import sys
sys.path.append("../src/")

import argparse
import contextlib
import math
import os
import re
import shutil
import socket
import subprocess
import zipfile
import gc
from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer

import h5py
import numpy as np
from numpy.lib import format as np_format
import torch
import torch.distributed as dist
from scipy.stats import linregress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler

try:
    from torch.distributed.algorithms.join import Join
except ImportError:  # pragma: no cover - only for very old PyTorch versions
    Join = None

from model_euler_ddp import FNO2d
from utilities3_euler_ddp import LpLoss, count_params
from utilities3_grain_ddp import log_normalize_np, reference_normalize_np


# -----------------------------------------------------------------------------
# Distributed/DDP helpers
# -----------------------------------------------------------------------------

def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _first_host_from_slurm_nodelist(nodelist):
    """Return the first hostname in SLURM_NODELIST.

    Prefer `scontrol show hostnames`, which handles all Slurm nodelist
    syntaxes.  A small regex fallback handles common forms such as
    `node[001-004]` when scontrol is unavailable.
    """
    if not nodelist:
        return socket.gethostname()

    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
            stderr=subprocess.DEVNULL,)
        hosts = [line.strip() for line in output.splitlines() if line.strip()]
        if hosts:
            return hosts[0]
    except Exception:
        pass

    match = re.match(r"^(?P<prefix>[^\[]+)\[(?P<body>[^\]]+)\]", nodelist)
    if match:
        prefix = match.group("prefix")
        first_group = match.group("body").split(",", 1)[0]
        first_index = first_group.split("-", 1)[0]
        return f"{prefix}{first_index}"

    return nodelist.split(",", 1)[0]


def _resolve_host_to_ipv4(hostname):
    """Resolve a hostname to an IPv4 address when possible.
    Returning an IPv4 literal keeps torch.distributed on
    AF_INET and avoids noisy `Address family not supported by protocol` warnings.
    """
    if not hostname:
        return hostname

    try:
        socket.inet_aton(hostname)
        if hostname.count(".") == 3:
            return hostname
    except OSError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in infos:
            addr = info[4][0]
            if addr and not addr.startswith("127."):
                return addr
    except OSError:
        pass

    return hostname


def _default_route_interface():
    """Return the Linux default-route network interface, or None if unavailable."""
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                fields = line.split()
                if len(fields) >= 2 and fields[1] == "00000000":
                    iface = fields[0]
                    if iface != "lo":
                        return iface
    except Exception:
        pass
    return None


def configure_gloo_ipv4_environment():
    """Prefer IPv4 for CPU/Gloo DDP."""
    if os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = _resolve_host_to_ipv4(os.environ["MASTER_ADDR"])

    if not os.environ.get("GLOO_SOCKET_IFNAME"):
        iface = _default_route_interface()
        if iface:
            os.environ["GLOO_SOCKET_IFNAME"] = iface


def configure_slurm_environment_for_ddp():
    """Translate Slurm's rank variables into torch.distributed env vars.
    """
    if _env_int("WORLD_SIZE", 1) > 1:
        return

    slurm_ntasks = _env_int("SLURM_NTASKS", 1)
    if slurm_ntasks <= 1:
        return

    slurm_rank = _env_int("SLURM_PROCID", 0)
    slurm_local_rank = _env_int("SLURM_LOCALID", 0)

    if not os.environ.get("WORLD_SIZE"):
        os.environ["WORLD_SIZE"] = str(slurm_ntasks)
    if not os.environ.get("RANK"):
        os.environ["RANK"] = str(slurm_rank)
    if not os.environ.get("LOCAL_RANK"):
        os.environ["LOCAL_RANK"] = str(slurm_local_rank)

    if "LOCAL_WORLD_SIZE" not in os.environ:
        tasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if tasks_per_node:
            m = re.match(r"(\d+)", tasks_per_node)
            os.environ["LOCAL_WORLD_SIZE"] = m.group(1) if m else "1"
        else:
            os.environ["LOCAL_WORLD_SIZE"] = "1"

    if not os.environ.get("MASTER_ADDR"):
        master_host = _first_host_from_slurm_nodelist(os.environ.get("SLURM_NODELIST"))
        os.environ["MASTER_ADDR"] = _resolve_host_to_ipv4(master_host)

    if not os.environ.get("MASTER_PORT"):
        job_id_text = os.environ.get("SLURM_JOB_ID", "0")
        digits = "".join(ch for ch in job_id_text if ch.isdigit())
        job_suffix = int(digits[-4:]) if digits else 0
        os.environ["MASTER_PORT"] = str(20000 + job_suffix)


def _open_h5(path, mode):
    """Open an HDF5 file with a safe fallback for file-lock-sensitive clusters."""
    try:
        return h5py.File(path, mode, locking=False)
    except (TypeError, ValueError):
        return h5py.File(path, mode)


def configure_cpu_threading_from_slurm():
    """Respect Slurm CPU allocation for CPU DDP jobs.
    """
    cpus_per_task = _env_int("SLURM_CPUS_PER_TASK", None)
    if cpus_per_task is None or cpus_per_task <= 0:
        return None

    n_threads = max(1, int(cpus_per_task))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, n_threads)))
    except RuntimeError:
        pass
    return n_threads


def setup_distributed(backend=None):
    """Initialize torch.distributed for torchrun or plain Slurm `srun`.
    """
    configure_slurm_environment_for_ddp()

    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    initialized_here = False

    distributed = world_size > 1 or (dist.is_available() and dist.is_initialized())

    if distributed and not (dist.is_available() and dist.is_initialized()):
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "gloo":
            configure_gloo_ipv4_environment()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return {
        "enabled": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "backend": backend,
        "initialized_here": initialized_here,}


def cleanup_distributed(distributed_context):
    if (
        distributed_context
        and distributed_context.get("initialized_here", False)
        and dist.is_available()
        and dist.is_initialized()):
        dist.destroy_process_group()


def rank_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def distributed_barrier(enabled):
    if enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


class DistributedNoPaddingSampler(Sampler):
    """Shard a map-style dataset across ranks without padding or duplication.
    """

    def __init__(self, dataset, rank, world_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank={self.rank} must be in [0, {self.world_size})")

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=generator).tolist()
        else:
            indices = list(range(n))
        return iter(indices[self.rank::self.world_size])

    def __len__(self):
        n = len(self.dataset)
        return (n + self.world_size - 1 - self.rank) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model


def assert_no_complex_parameters_for_cpu_ddp(model, device, distributed):
    if not distributed or device.type != "cpu":
        return
    complex_names = [
        name for name, param in model.named_parameters()
        if torch.is_complex(param)
    ]
    if complex_names:
        preview = ", ".join(complex_names[:8])
        raise RuntimeError(
            "CPU/Gloo DDP cannot synchronize complex-valued nn.Parameter tensors "
            f"in this PyTorch build. Complex parameters found: {preview}. "
            "Use the patched model_euler.py, which stores Fourier weights as "
            "real-valued [real, imag] parameters.")


def sync_model_state_from_rank0(model, distributed):
    """Make rank 0's parameters and buffers canonical before validation/save."""
    if not distributed:
        return
    module = unwrap_ddp(model)
    with torch.no_grad():
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


def unpack_batch(batch):
    if len(batch) == 6:
        xx, yy, C1, C2, C3, sample_index = batch
    elif len(batch) == 5:
        xx, yy, C1, C2, C3 = batch
        sample_index = None
    else:
        raise ValueError(f"Expected a 5- or 6-item batch, got {len(batch)} items")
    return xx, yy, C1, C2, C3, sample_index


def gather_epoch_losses_to_rank0(local_losses, local_indices, device, rank, world_size, distributed):
    """Gather per-sample loss values from all ranks and return them on rank 0.
    This gathers loss scalars and sample indices rather than raw fields/images;
    that is equivalent for epoch loss computation and avoids transferring the
    full HDF5 samples at every epoch end.
    """
    if local_losses:
        local_loss = torch.cat([x.detach().reshape(-1).to(device=device, dtype=torch.float64) for x in local_losses])
    else:
        local_loss = torch.empty(0, device=device, dtype=torch.float64)

    if local_indices:
        local_index = torch.cat([x.detach().reshape(-1).to(device=device, dtype=torch.long) for x in local_indices])
    else:
        local_index = torch.arange(local_loss.numel(), device=device, dtype=torch.long)

    if local_loss.numel() != local_index.numel():
        raise ValueError(
            f"local_loss and local_index length mismatch: {local_loss.numel()} vs {local_index.numel()}")

    if not distributed:
        return local_loss.cpu(), local_index.cpu()

    local_len = torch.tensor([local_loss.numel()], device=device, dtype=torch.long)
    length_tensors = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(length_tensors, local_len)
    lengths = [int(x.item()) for x in length_tensors]
    max_len = max(lengths) if lengths else 0

    if max_len == 0:
        if rank == 0:
            return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.long)
        return None, None

    padded_loss = torch.zeros(max_len, device=device, dtype=torch.float64)
    padded_index = torch.full((max_len,), -1, device=device, dtype=torch.long)
    n_local = int(local_len.item())
    if n_local > 0:
        padded_loss[:n_local] = local_loss
        padded_index[:n_local] = local_index

    gathered_losses = [torch.empty_like(padded_loss) for _ in range(world_size)]
    gathered_indices = [torch.empty_like(padded_index) for _ in range(world_size)]
    dist.all_gather(gathered_losses, padded_loss)
    dist.all_gather(gathered_indices, padded_index)

    if rank != 0:
        return None, None

    losses = []
    indices = []
    for rank_i, n_i in enumerate(lengths):
        if n_i > 0:
            losses.append(gathered_losses[rank_i][:n_i].cpu())
            indices.append(gathered_indices[rank_i][:n_i].cpu())

    if not losses:
        return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.long)
    return torch.cat(losses), torch.cat(indices)


def mean_unique_loss(losses, indices):
    if losses is None or losses.numel() == 0:
        raise RuntimeError("No losses were gathered for this epoch.")

    seen = set()
    keep = []
    for pos, sample_id in enumerate(indices.tolist()):
        if sample_id < 0:
            continue
        if sample_id not in seen:
            seen.add(sample_id)
            keep.append(pos)

    if keep and len(keep) != losses.numel():
        losses = losses[torch.tensor(keep, dtype=torch.long)]

    return float(losses.double().mean().item())


def broadcast_rank0_float(value, device, distributed):
    if value is None:
        value = 0.0
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if distributed:
        dist.broadcast(tensor, src=0)
    return float(tensor.item())


class H5Dataset(Dataset):
    DATASET_NAMES = ("euler1_known", "euler1_predict", "strain_rate", "temperature", "pressure")

    def __init__(self, h5_file_path, return_index=False):
        super().__init__()
        self.h5_file_path = str(h5_file_path)
        self.return_index = bool(return_index)

        with _open_h5(self.h5_file_path, "r") as f:
            missing = [name for name in self.DATASET_NAMES if name not in f]
            if missing:
                raise KeyError(f"{self.h5_file_path} is missing datasets: {missing}")

            self.N = int(f["euler1_known"].shape[0])
            for name in self.DATASET_NAMES[1:]:
                if int(f[name].shape[0]) != self.N:
                    raise ValueError(
                        f"Inconsistent sample count in {self.h5_file_path}: "
                        f"euler1_known has {self.N}, {name} has {f[name].shape[0]}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        with _open_h5(self.h5_file_path, "r") as f:
            euler1_known = torch.from_numpy(f["euler1_known"][idx]).float()
            euler1_predict = torch.from_numpy(f["euler1_predict"][idx]).float()
            strain_rate = torch.from_numpy(f["strain_rate"][idx]).float()
            temperature = torch.from_numpy(f["temperature"][idx]).float()
            pressure = torch.from_numpy(f["pressure"][idx]).float()

        if self.return_index:
            return euler1_known, euler1_predict, strain_rate, temperature, pressure, torch.tensor(int(idx), dtype=torch.long)

        return euler1_known, euler1_predict, strain_rate, temperature, pressure


class CircularAngleLoss:
    def __init__(self, size_average=True, reduction=True):
        self.size_average = size_average
        self.reduction = reduction

    def __call__(self, x, y):
        if x.shape != y.shape:
            raise ValueError(f"Loss inputs must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")

        num_examples = x.shape[0]
        diff_rad = torch.atan2(
            torch.sin(math.pi * (x - y)),
            torch.cos(math.pi * (x - y)),)
        losses = torch.mean(diff_rad.reshape(num_examples, -1) ** 2, dim=1)

        if not self.reduction:
            return losses
        if self.size_average:
            return torch.mean(losses)
        return torch.sum(losses)


class RootMeanSquaredErrorLoss:
    def __init__(self, size_average=True, reduction=True, eps=1e-12):
        self.size_average = size_average
        self.reduction = reduction
        self.eps = float(eps)

    def __call__(self, x, y):
        if x.shape != y.shape:
            raise ValueError(f"Loss inputs must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")

        num_examples = x.shape[0]
        mse_per_sample = torch.mean((x - y).reshape(num_examples, -1) ** 2, dim=1)
        losses = torch.sqrt(mse_per_sample + self.eps)

        if not self.reduction:
            return losses
        if self.size_average:
            return torch.mean(losses)
        return torch.sum(losses)


def build_loss(loss_func, batch_average=True, reduction=True):
    loss_name = str(loss_func).lower().replace("-", "_")
    if loss_name in {"circular", "circular_mse", "angle", "angular"}:
        return CircularAngleLoss(size_average=batch_average, reduction=reduction)
    if loss_name in {"mean_square_root", "rmse", "root_mse", "root_mean_square", "root_mean_squared", "root_mean_square_error", "sqrt_mse", "msr"}:
        return RootMeanSquaredErrorLoss(size_average=batch_average, reduction=reduction)
    if loss_name in {"l2", "lp", "relative_l2", "rel_l2"}:
        return LpLoss(size_average=batch_average, reduction=reduction)
    raise ValueError(
        "loss_func must be one of: circular_mse, rmse, mean_square_root, or L2")


def summarize_hdf5_dataset(h5_file_path, dataset_name, sample_count=16):
    with _open_h5(h5_file_path, "r") as f:
        dset = f[dataset_name]
        shape = dset.shape
        dtype = dset.dtype
        print(f"{dataset_name} shape: {shape}, dtype: {dtype}")

        N = int(shape[0])
        if N == 0:
            print("  dataset is empty\n")
            return

        sample_count = max(1, min(int(sample_count), N))
        indices = np.linspace(0, N - 1, sample_count, dtype=int)

        global_min = np.inf
        global_max = -np.inf
        for index in indices:
            chunk_data = dset[index]
            local_min = float(np.nanmin(chunk_data))
            local_max = float(np.nanmax(chunk_data))
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

        print(f"  sampled data range over {sample_count} samples: [{global_min}, {global_max}]\n")


def _list_euler_npz_files(data_path):
    files = sorted(
        f for f in os.listdir(data_path)
        if f.endswith(".npz") and f.startswith("euler_1"))
    if not files:
        raise FileNotFoundError(f"No euler_1*.npz files found in {data_path}")
    return files


def _window_count(num_timesteps, step_known, step_predict, step_size):
    """Number of valid windows using step_size as a target offset.
    """
    num_timesteps = int(num_timesteps)
    step_known = int(step_known)
    step_predict = int(step_predict)
    step_size = int(step_size)

    if step_known <= 0 or step_predict <= 0:
        raise ValueError(f"step_known and step_predict must be positive, got {step_known}, {step_predict}.")
    if step_size < step_known:
        raise ValueError(
            "step_size is now interpreted as the target offset from the first known frame. "
            f"For future prediction it must be >= step_known; got step_size={step_size}, "
            f"step_known={step_known}.")

    last_known_start = num_timesteps - step_known
    last_pred_start = num_timesteps - step_size - step_predict
    return int(min(last_known_start, last_pred_start) + 1)


def _npz_arr0_member_name(npz_path):
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = zf.namelist()
    for candidate in ("arr_0.npy", "arr_0"):
        if candidate in names:
            return candidate
    npy_names = [name for name in names if name.endswith(".npy")]
    if len(npy_names) == 1:
        return npy_names[0]
    raise KeyError(f"Could not find arr_0.npy in {npz_path}; archive members are {names[:10]}")


def _npz_arr0_shape_dtype(npz_path):
    member = _npz_arr0_member_name(npz_path)
    with zipfile.ZipFile(npz_path, "r") as zf:
        with zf.open(member, "r") as fp:
            version = np_format.read_magic(fp)
            if version == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(fp)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np_format.read_array_header_2_0(fp)
            else:
                raise ValueError(f"Unsupported .npy version {version} inside {npz_path}")
    if fortran_order:
        raise ValueError(f"Fortran-order arr_0 arrays are not supported in {npz_path}")
    return tuple(int(x) for x in shape), np.dtype(dtype)


def _default_npz_extract_dir(save_traindata_path):
    for env_name in ("SLURM_TMPDIR", "LOCAL_SCRATCH", "TMPDIR"):
        value = os.environ.get(env_name)
        if value and not value.startswith("/dev/shm"):
            return value
    return os.path.join(str(save_traindata_path), "_npz_extract_tmp")


@contextmanager
def _open_npz_arr0_as_memmap(npz_path, extract_dir=None):
    """arr_0 as a read-only array without keeping the full .npz in RAM.
    """
    npz_path = str(npz_path)
    extract_dir = _default_npz_extract_dir(Path(npz_path).parent) if extract_dir is None else str(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    member = _npz_arr0_member_name(npz_path)
    stem = Path(npz_path).stem
    tmp_path = os.path.join(extract_dir, f"{stem}.arr_0.pid{os.getpid()}.npy")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    arr = None
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

        arr = np.load(tmp_path, mmap_mode="r")
        yield arr
    finally:
        try:
            del arr
        except UnboundLocalError:
            pass
        gc.collect()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _array_nbytes(shape, dtype):
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def _format_bytes(nbytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _inspect_npz_files(data_path, euler1_files, grid_size, step_known, step_predict, step_size):
    metadata = []
    inferred_grid_size = grid_size

    for euler1_file in euler1_files:
        euler1_path = os.path.join(data_path, euler1_file)
        shape, dtype = _npz_arr0_shape_dtype(euler1_path)

        if len(shape) != 4 or shape[0] < 4:
            raise ValueError(
                f"{euler1_path} must have shape [>=4, X, Y, T]; got {shape}")

        _, nx, ny, nt = shape
        if nx != ny:
            raise ValueError(f"Only square grids are supported; {euler1_path} has {nx} x {ny}")

        if inferred_grid_size is None:
            inferred_grid_size = int(nx)
        elif int(inferred_grid_size) != int(nx):
            raise ValueError(
                f"model_dimension/grid_size={inferred_grid_size} but {euler1_path} has grid size {nx}. "
                "Pass the correct --model_dimension or omit it to infer from data.")

        n_windows = _window_count(nt, step_known, step_predict, step_size)
        if n_windows <= 0:
            raise ValueError(
                f"{euler1_path} has only {nt} timesteps, which is not enough for "
                f"step_known={step_known}, step_size={step_size}, step_predict={step_predict}.")

        metadata.append(
            {"file": euler1_file,
            "path": euler1_path,
            "shape": tuple(int(v) for v in shape),
            "dtype": str(dtype),
            "nbytes": _array_nbytes(shape, dtype),
            "n_windows": int(n_windows),})

    return metadata, int(inferred_grid_size)


def _split_files_by_simulation(metadata, train_ratio=0.8, valid_ratio=0.1, seed=12345):
    n_files = len(metadata)
    if n_files < 3:
        raise ValueError(
            "Need at least 3 euler_1*.npz simulation files for a leakage-free "
            "train/validation/test split. Add more files or create external splits.")

    rng = np.random.default_rng(seed)
    shuffled = list(metadata)
    rng.shuffle(shuffled)

    n_train = max(1, int(math.floor(train_ratio * n_files)))
    n_valid = max(1, int(math.floor(valid_ratio * n_files)))
    n_test = n_files - n_train - n_valid

    if n_test < 1:
        n_test = 1
        if n_train >= n_valid and n_train > 1:
            n_train -= 1
        elif n_valid > 1:
            n_valid -= 1
        else:
            raise ValueError("Could not create non-empty train/valid/test file-level splits.")

    train_meta = shuffled[:n_train]
    valid_meta = shuffled[n_train:n_train + n_valid]
    test_meta = shuffled[n_train + n_valid:]

    return {"train": train_meta, "valid": valid_meta, "test": test_meta}


def _check_normalized_range(name, array, atol=1e-5):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf after normalization.")
    amin = float(np.min(array))
    amax = float(np.max(array))
    if amin < -1.0 - atol or amax > 1.0 + atol:
        raise ValueError(
            f"{name} is outside [-1, 1] after normalization: min={amin}, max={amax}. "
            "Check the reference bounds.")


def _make_normalized_sample(
    euler1_data,
    j,
    step_known,
    step_predict,
    step_size,
    T_ref_bound,
    H_ref_bound,
    S_ref_bound,
):
    known_slice = euler1_data[0, :, :, j:j + step_known].astype(np.float32) / 180.0

    # step_size is the direct target offset from the first known frame.
    pred_start = j + step_size
    predict_slice = euler1_data[0, :, :, pred_start:pred_start + step_predict].astype(np.float32) / 180.0

    sr_slice = euler1_data[1, :, :, j:j + step_known].astype(np.float32)
    if np.any(sr_slice <= 0):
        raise ValueError("strain_rate contains non-positive values; log10 normalization is invalid.")
    sr_slice = log_normalize_np(sr_slice, min_ref=S_ref_bound[0], max_ref=S_ref_bound[1]).astype(np.float32)

    pr_slice = euler1_data[2, :, :, j:j + step_known].astype(np.float32)
    pr_slice = pr_slice + np.float32(900.0 * 9.8)
    pr_slice = reference_normalize_np(pr_slice, min_ref=H_ref_bound[0], max_ref=H_ref_bound[1]).astype(np.float32)

    # temperature preprocessing gives higher temperature more numerical leverage while keeping the transformed values
    # inside [-1, 1] for the present data/reference range.
    temp_slice = euler1_data[3, :, :, j:j + step_known].astype(np.float32)
    temp_slice = 1 / temp_slice
    # temp_slice = reference_normalize_np(temp_slice, min_ref=T_ref_bound[0], max_ref=T_ref_bound[1]).astype(np.float32)

    _check_normalized_range("euler1_known", known_slice)
    _check_normalized_range("euler1_predict", predict_slice)
    _check_normalized_range("strain_rate", sr_slice)
    _check_normalized_range("pressure", pr_slice)
    _check_normalized_range("temperature", temp_slice)

    return known_slice, predict_slice, sr_slice, temp_slice, pr_slice


def _write_split_h5(
    split_name,
    split_metadata,
    h5_file_path,
    grid_size,
    step_known,
    step_predict,
    step_size,
    T_ref_bound,
    H_ref_bound,
    S_ref_bound,
    npz_extract_dir=None,
):
    total_samples = int(sum(item["n_windows"] for item in split_metadata))
    if total_samples <= 0:
        raise ValueError(f"{split_name} split has no training windows.")

    h5_file_path = str(h5_file_path)
    Path(h5_file_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_h5_file_path = f"{h5_file_path}.tmp.pid{os.getpid()}"
    if os.path.exists(tmp_h5_file_path):
        os.remove(tmp_h5_file_path)

    if npz_extract_dir is None:
        npz_extract_dir = _default_npz_extract_dir(Path(h5_file_path).parent)
    print(f"Writing {split_name} split: {len(split_metadata)} files, {total_samples} samples -> {h5_file_path}")
    print(f"  streaming .npz arrays through memmapped temporary .npy files in: {npz_extract_dir}")

    try:
        with _open_h5(tmp_h5_file_path, "w") as f:
            dsets = {
                "euler1_known": f.create_dataset(
                    "euler1_known",
                    shape=(total_samples, grid_size, grid_size, step_known),
                    dtype=np.float32,
                    chunks=(1, grid_size, grid_size, step_known),),
                "euler1_predict": f.create_dataset(
                    "euler1_predict",
                    shape=(total_samples, grid_size, grid_size, step_predict),
                    dtype=np.float32,
                    chunks=(1, grid_size, grid_size, step_predict),),
                "strain_rate": f.create_dataset(
                    "strain_rate",
                    shape=(total_samples, grid_size, grid_size, step_known),
                    dtype=np.float32,
                    chunks=(1, grid_size, grid_size, step_known),),
                "temperature": f.create_dataset(
                    "temperature",
                    shape=(total_samples, grid_size, grid_size, step_known),
                    dtype=np.float32,
                    chunks=(1, grid_size, grid_size, step_known),),
                "pressure": f.create_dataset(
                    "pressure",
                    shape=(total_samples, grid_size, grid_size, step_known),
                    dtype=np.float32,
                    chunks=(1, grid_size, grid_size, step_known),),
            }

            f.attrs["split_name"] = split_name
            f.attrs["source_files"] = np.array([item["file"] for item in split_metadata], dtype=h5py.string_dtype())
            f.attrs["step_known"] = step_known
            f.attrs["step_predict"] = step_predict
            f.attrs["step_size"] = step_size

            offset = 0
            for item in split_metadata:
                print(
                    f"  {split_name}: streaming {item['file']}, shape={item['shape']}, "
                    f"dtype={item.get('dtype', 'unknown')}, on-disk array size ~ {_format_bytes(item.get('nbytes', 0))}, "
                    f"windows={item['n_windows']}")

                with _open_npz_arr0_as_memmap(item["path"], extract_dir=npz_extract_dir) as euler1_data:
                    for j in range(item["n_windows"]):
                        known, predict, sr, temp, pressure = _make_normalized_sample(
                            euler1_data,
                            j,
                            step_known,
                            step_predict,
                            step_size,
                            T_ref_bound,
                            H_ref_bound,
                            S_ref_bound,)

                        row = offset + j
                        dsets["euler1_known"][row] = known
                        dsets["euler1_predict"][row] = predict
                        dsets["strain_rate"][row] = sr
                        dsets["temperature"][row] = temp
                        dsets["pressure"][row] = pressure

                offset += item["n_windows"]

        os.replace(tmp_h5_file_path, h5_file_path)
    finally:
        if os.path.exists(tmp_h5_file_path):
            os.remove(tmp_h5_file_path)

    print(f"Saved {split_name} data to {h5_file_path}\n")
    return h5_file_path

def _split_file_names(splits):
    return {
        name: [item["file"] for item in split_metadata]
        for name, split_metadata in splits.items()}


def load_data(
    data_path,
    save_testdata_path,
    save_traindata_path,
    T_ref_bound,
    H_ref_bound,
    kde_ref_bound,
    S_ref_bound,
    num_input_params=4,
    num_timesteps=24,
    grid_size=None,
    step_known=1,
    step_predict=1,
    step_size=499,
    batch_size=2,
    shuffle=True,
    split_seed=12345,
    num_workers=2,
    rank=0,
    world_size=1,
    distributed=False,
    npz_extract_dir=None,
    reuse_existing_h5=False,
):
    """Load .npz files, create leakage-free HDF5 splits, and return DataLoaders."""

    del kde_ref_bound, num_input_params, num_timesteps  # kept for backward-compatible call signature

    data_path = str(data_path)
    save_traindata_path = str(save_traindata_path)
    save_testdata_path = str(save_testdata_path)

    euler1_files = _list_euler_npz_files(data_path)
    rank_print(rank, "-----------------------------------")
    rank_print(rank, f"Number of npz files to read: {len(euler1_files)}")
    rank_print(rank, "-----------------------------------")

    num_npz_files = len(euler1_files)
    metadata = None
    splits = None

    # Rank 0 is the only process that reads the large .npz arrays and writes
    # HDF5 split files.  Other ranks wait at the barrier below, then read the
    # finished HDF5 files through their own sharded DataLoader.
    # if rank == 0:
    #     metadata, grid_size = _inspect_npz_files(
    #         data_path,
    #         euler1_files,
    #         grid_size,
    #         step_known,
    #         step_predict,
    #         step_size,
    #     )

    #     splits = _split_files_by_simulation(metadata, train_ratio=0.8, valid_ratio=0.1, seed=split_seed)
    #     print("File-level split to avoid sliding-window leakage:")
    #     for split_name, files in _split_file_names(splits).items():
    #         n_samples = sum(item["n_windows"] for item in splits[split_name])
    #         print(f"  {split_name}: {len(files)} files, {n_samples} samples")
    #         print(f"    {files}")

    # Include windowing parameters in the generated HDF5 names.  
    h5_tag = f"known{step_known}_Sadaptive_pred{step_predict}"
    train_h5 = os.path.join(save_traindata_path, f"jcp_0602_syn_euler1_{h5_tag}_train_data.h5")
    valid_h5 = os.path.join(save_traindata_path, f"jcp_0602_syn_euler1_{h5_tag}_valid_data.h5")
    test_h5 = os.path.join(save_testdata_path, f"jcp_0602_syn_euler1_{h5_tag}_test_data.h5")

    if rank == 0:
    #     have_existing_h5 = all(os.path.exists(path) for path in (train_h5, valid_h5, test_h5))
    #     if reuse_existing_h5 and have_existing_h5:
    #         print("Reusing existing HDF5 split files because --reuse_existing_h5 was set:")
    #         print(f"  train: {train_h5}")
    #         print(f"  valid: {valid_h5}")
    #         print(f"  test : {test_h5}")
    #     else:
    #         _write_split_h5(
    #             "train",
    #             splits["train"],
    #             train_h5,
    #             grid_size,
    #             step_known,
    #             step_predict,
    #             step_size,
    #             T_ref_bound,
    #             H_ref_bound,
    #             S_ref_bound,
    #             npz_extract_dir=npz_extract_dir,
    #         )
    #         _write_split_h5(
    #             "valid",
    #             splits["valid"],
    #             valid_h5,
    #             grid_size,
    #             step_known,
    #             step_predict,
    #             step_size,
    #             T_ref_bound,
    #             H_ref_bound,
    #             S_ref_bound,
    #             npz_extract_dir=npz_extract_dir,
    #         )
    #         _write_split_h5(
    #             "test",
    #             splits["test"],
    #             test_h5,
    #             grid_size,
    #             step_known,
    #             step_predict,
    #             step_size,
    #             T_ref_bound,
    #             H_ref_bound,
    #             S_ref_bound,
    #             npz_extract_dir=npz_extract_dir,
    #         )

        for name in H5Dataset.DATASET_NAMES:
            summarize_hdf5_dataset(train_h5, name, sample_count=16)
        for name in H5Dataset.DATASET_NAMES:
            summarize_hdf5_dataset(valid_h5, name, sample_count=16)

    # All nonzero ranks wait until rank 0 has finished writing the HDF5 files.
    distributed_barrier(distributed)

    train_dataset = H5Dataset(train_h5, return_index=True)
    valid_dataset = H5Dataset(valid_h5, return_index=True)

    if distributed:
        train_sampler = DistributedNoPaddingSampler(
            train_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=split_seed,
        )
        valid_sampler = DistributedNoPaddingSampler(
            valid_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            seed=split_seed,
        )
        loader_shuffle = False
    else:
        train_sampler = None
        valid_sampler = None
        loader_shuffle = shuffle

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=train_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if distributed:
        rank_print(
            rank,
            f"DDP data sharding: world_size={world_size}; "
            f"rank 0 train samples per epoch={len(train_sampler)}; "
            "all ranks use disjoint no-padding shards.")

    return train_loader, valid_loader, num_npz_files


def _grad_norm(model):
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_sq += float(param.grad.detach().norm(2).item() ** 2)
    return math.sqrt(total_sq)



def _compute_recurrent_loss_vector(model, xx, yy, C1, C2, C3, myloss, T_in, T_end, step, ep=None):
    """Return one accumulated loss scalar per sample in the local batch."""
    loss_vector = None

    for t in range(0, (T_end - T_in), step):
        y = yy[..., t:t + step]
        pred = model(xx, C1, C2, C3)

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            where = "" if ep is None else f" at epoch {ep}, step {t}"
            raise FloatingPointError(f"NaN/Inf in predictions{where}")

        b = xx.size(0)
        step_loss = myloss(pred.reshape(b, -1), y.reshape(b, -1))
        if step_loss.ndim == 0:
            step_loss = step_loss.repeat(b)
        else:
            step_loss = step_loss.reshape(b)

        if loss_vector is None:
            loss_vector = torch.zeros_like(step_loss)
        loss_vector = loss_vector + step_loss

        xx = torch.cat((xx[..., step:], pred), dim=-1)

    if loss_vector is None:
        raise RuntimeError("No recurrent prediction steps were executed. Check T_in, T_end, and step.")
    return loss_vector


def train(
    save_model_file,
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    myloss,
    costFunc,
    epoch,
    device,
    T_in,
    T_end,
    step,
    batch_size,
    smoothness_weight,
    patience_window,
    patience_ratio,
    band_tolerance,
    scheduler_gamma,
    rank=0,
    world_size=1,
    distributed=False,
):
    del costFunc, batch_size, smoothness_weight, patience_ratio, band_tolerance, scheduler_gamma

    rank_print(rank, f"\nThe model has {count_params(unwrap_ddp(model))} trainable parameters\n")
    rank_print(rank, f"Training the model on {device} for {epoch} epochs ...\n")

    if distributed and Join is None:
        raise RuntimeError(
            "This DDP implementation uses a no-padding sampler and requires "
            "torch.distributed.algorithms.join.Join. Please use a PyTorch version "
            "that provides Join, or switch to a padded sampler.")

    train_loss = torch.zeros(epoch)
    valid_loss = torch.zeros(epoch)

    loss_history = []
    slope = 0.0

    for ep in range(epoch):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(ep)

        model.train()
        t1 = default_timer()
        local_train_losses = []
        local_train_indices = []
        grad_norm_before = 0.0
        grad_norm_after = 0.0
        local_train_batches = 0

        train_context = Join([model]) if distributed else contextlib.nullcontext()
        with train_context:
            for batch in train_loader:
                xx, yy, C1, C2, C3, sample_index = unpack_batch(batch)
                xx = xx.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                C1 = C1.to(device, non_blocking=True)
                C2 = C2.to(device, non_blocking=True)
                C3 = C3.to(device, non_blocking=True)
                if sample_index is None:
                    sample_index = torch.arange(xx.size(0), dtype=torch.long)
                sample_index = sample_index.to(device, non_blocking=True)

                loss_vector = _compute_recurrent_loss_vector(
                    model,
                    xx,
                    yy,
                    C1,
                    C2,
                    C3,
                    myloss,
                    T_in,
                    T_end,
                    step,
                    ep=ep,)
                loss = loss_vector.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm_before = _grad_norm(model)
                if grad_norm_before > 10.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                grad_norm_after = _grad_norm(model)

                optimizer.step()

                local_train_losses.append(loss_vector.detach())
                local_train_indices.append(sample_index.detach())
                local_train_batches += 1

        train_losses_cpu, train_indices_cpu = gather_epoch_losses_to_rank0(
            local_train_losses,
            local_train_indices,
            device,
            rank,
            world_size,
            distributed,)
        if rank == 0:
            train_epoch_loss_value = mean_unique_loss(train_losses_cpu, train_indices_cpu)
        else:
            train_epoch_loss_value = None
        train_epoch_loss_value = broadcast_rank0_float(train_epoch_loss_value, device, distributed)
        train_loss[ep] = train_epoch_loss_value

        # Synchronize rank 0's canonical parameters/buffers before distributed validation. 
        sync_model_state_from_rank0(model, distributed)

        # Use the unwrapped module in eval mode to avoid DDP forward collectives
        # when ranks have slightly different validation batch counts.
        eval_model = unwrap_ddp(model)
        eval_model.eval()
        local_valid_losses = []
        local_valid_indices = []
        local_valid_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                xx, yy, C1, C2, C3, sample_index = unpack_batch(batch)
                xx = xx.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                C1 = C1.to(device, non_blocking=True)
                C2 = C2.to(device, non_blocking=True)
                C3 = C3.to(device, non_blocking=True)
                if sample_index is None:
                    sample_index = torch.arange(xx.size(0), dtype=torch.long)
                sample_index = sample_index.to(device, non_blocking=True)

                loss_vector = _compute_recurrent_loss_vector(
                    eval_model,
                    xx,
                    yy,
                    C1,
                    C2,
                    C3,
                    myloss,
                    T_in,
                    T_end,
                    step,
                    ep=None,)
                local_valid_losses.append(loss_vector.detach())
                local_valid_indices.append(sample_index.detach())
                local_valid_batches += 1

        valid_losses_cpu, valid_indices_cpu = gather_epoch_losses_to_rank0(
            local_valid_losses,
            local_valid_indices,
            device,
            rank,
            world_size,
            distributed,)
        if rank == 0:
            valid_epoch_loss_value = mean_unique_loss(valid_losses_cpu, valid_indices_cpu)
        else:
            valid_epoch_loss_value = None
        valid_epoch_loss_value = broadcast_rank0_float(valid_epoch_loss_value, device, distributed)
        valid_loss[ep] = valid_epoch_loss_value

        if local_train_batches == 0 and not distributed:
            raise RuntimeError(
                "The training DataLoader produced zero batches. "
                "Check batch_size and dataset size.")
        if local_valid_batches == 0 and not distributed:
            raise RuntimeError("The validation DataLoader produced zero batches.")

        # All ranks step the scheduler using the same rank-0 global validation loss.
        if scheduler is not None:
            scheduler.step(valid_epoch_loss_value)

        if rank == 0:
            loss_history.append(valid_epoch_loss_value)
            if len(loss_history) > patience_window:
                loss_history.pop(0)
            if len(loss_history) == patience_window:
                log_loss = np.log(np.asarray(loss_history, dtype=np.float64))
                slope, _, _, _, _ = linregress(np.arange(patience_window), log_loss)

            t2 = default_timer()
            print(
                f"epoch {ep:04d}, time {(t2 - t1) / 60:.2f} min, "
                f"train loss {train_epoch_loss_value:.6f}, valid loss {valid_epoch_loss_value:.6f}, "
                f"grad norm before/after {grad_norm_before:.3f}/{grad_norm_after:.3f}, "
                f"lr {optimizer.param_groups[0]['lr']:.6g}, slope {slope:.4f}")
            torch.save(
                {'model': (model.module if isinstance(model, DDP) else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,},
                save_model_file )

    return train_loss, valid_loss


def main(
    data_path,
    save_path,
    save_testdata_path,
    save_traindata_path,
    model_dimension=None,
    epochs_input=20,
    converge_name="jcp_syn_euler1_loss_convergence",
    batch_size=320,
    learning_rate=0.001,
    mode1=24,
    mode2=24,
    width=16,
    activation_func="tanh",
    loss_func="mean_square_root",
    step_known=2,
    step_predict=1,
    step_size=499,
    split_seed=12345,
    num_workers=2,
    ddp_backend=None,
    npz_extract_dir=None,
    reuse_existing_h5=False,
):
    del converge_name

    configured_threads = configure_cpu_threading_from_slurm()
    distributed_context = setup_distributed(backend=ddp_backend)
    rank = distributed_context["rank"]
    world_size = distributed_context["world_size"]
    local_rank = distributed_context["local_rank"]
    distributed = distributed_context["enabled"]
    device = distributed_context["device"]

    if rank == 0 and configured_threads is not None:
        print(f"Using torch CPU threads per rank from SLURM_CPUS_PER_TASK: {configured_threads}")

    try:
        scheduler_step = 200
        scheduler_gamma = 0.8
        scheduler_threshold = 1e-4
        smoothness_weight = 10
        patience_ratio = 0.9
        patience_window = 100
        band_tolerance = 0.15

        # Per-sample losses are required so rank 0 can compute true epoch
        # losses after gathering all rank-local sample losses.
        batch_average = False

        epochs = int(epochs_input)
        num_input_params = 4
        num_timesteps = 25

        # Reference bounds: [low, high].
        T_ref_bound = np.array([-26.0, -1.0], dtype=np.float32)
        H_ref_bound = np.array([1 * 900 * 9.8, 1002 * 900 * 9.8], dtype=np.float32)
        grain_ref_bound = np.array([0.0, 1.0], dtype=np.float32)
        S_ref_bound = np.array([0.99999e-12, 1.60001e-8], dtype=np.float32)

        save_model_file = f"{save_path}_known{step_known}_offset{step_size}_m{mode1}x{mode2}_w{width}_pred{step_predict}_epoch{epochs}.pt"

        if distributed:
            rank_print(
                rank,
                f"DDP enabled: world_size={world_size}, backend={dist.get_backend()}, "
                f"device={device}. Launch one process per CPU worker/GPU with torchrun or srun.")
        else:
            rank_print(rank, f"DDP disabled: single-process training on {device}.")

        cpus_per_task = _env_int("SLURM_CPUS_PER_TASK", None)
        if rank == 0 and cpus_per_task is not None and num_workers >= max(1, cpus_per_task):
            print(
                f"Warning: --num_workers={num_workers} with SLURM_CPUS_PER_TASK={cpus_per_task} "
                "oversubscribes CPU cores. For --cpus-per-task=1, use --num_workers 0 "
                "or increase --cpus-per-task.")

        train_loader, valid_loader, num_npz_files = load_data(
            data_path,
            save_testdata_path,
            save_traindata_path,
            T_ref_bound,
            H_ref_bound,
            grain_ref_bound,
            S_ref_bound,
            num_input_params,
            num_timesteps,
            model_dimension,
            step_known,
            step_predict,
            step_size,
            batch_size,
            True,
            split_seed=split_seed,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            npz_extract_dir=npz_extract_dir,
            reuse_existing_h5=reuse_existing_h5,)

        rank_print(
            rank,
            "\nTraining using activation function:",
            activation_func,
            ", loss function:",
            loss_func,
            ", rank-0 epoch loss from gathered per-sample losses?: True",)
        rank_print(
            rank,
            "\nRecurring property: steps known", step_known,
            ", steps to predict:", step_predict,
            ", target offset step_size:", step_size,
        )
        rank_print(
            rank,
            f"Model parameters: mode1={mode1}, mode2={mode2}, width={width}, "
            f"per-rank batch size={batch_size}, base learning rate={learning_rate}")

        base_model = FNO2d(mode1, mode2, width, step_known, activation_func, loss_func).to(device)
        assert_no_complex_parameters_for_cpu_ddp(base_model, device, distributed)
        if distributed:
            if device.type == "cuda":
                model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            else:
                model = DDP(base_model, find_unused_parameters=False)
        else:
            model = base_model

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_gamma,
            patience=scheduler_step,
            threshold=scheduler_threshold,
            cooldown=0,
            min_lr=1e-5,)
        rank_print(
            rank,
            "Schedule params: ReduceLROnPlateau, "
            f"patience={scheduler_step}, threshold={scheduler_threshold}, factor={scheduler_gamma}")

        myloss = build_loss(loss_func, batch_average=batch_average, reduction=False)
        costFunc = torch.nn.MSELoss(reduction="sum")

        train_loss, valid_loss = train(
            save_model_file,
            model,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            myloss,
            costFunc,
            epochs,
            device,
            step_known,
            step_known + step_predict,
            1,
            batch_size,
            smoothness_weight,
            patience_window,
            patience_ratio,
            band_tolerance,
            scheduler_gamma,
            rank=rank,
            world_size=world_size,
            distributed=distributed,)

        sync_model_state_from_rank0(model, distributed)
        distributed_barrier(distributed)

        if rank == 0:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            loss_tag = f"known{step_known}_offset{step_size}_m{mode1}x{mode2}_w{width}_pred{step_predict}_N{num_npz_files}_epoch{epochs}"
            train_loss_file = f"{save_path}_{loss_tag}_train_loss.pt"
            valid_loss_file = f"{save_path}_{loss_tag}_valid_loss.pt"

            torch.save(unwrap_ddp(model).state_dict(), save_model_file)
            torch.save(train_loss, train_loss_file)
            torch.save(valid_loss, valid_loss_file)

            print("Model saved to:", save_model_file)
            print("Train loss saved to:", train_loss_file)
            print("Valid loss saved to:", valid_loss_file)

        distributed_barrier(distributed)

    finally:
        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Euler1 FNO model with optional DDP")
    parser.add_argument("--data_path", type=str, default="/synthetic/data_for_FNO_training/train_data/", help="Path to euler_1*.npz data")
    parser.add_argument("--save_path", type=str, default="./../model/model", help="Base path/name for saved model and loss tensors")
    parser.add_argument("--save_testdata_path", type=str, default="/data/test/", help="Directory for saved test HDF5 data")
    parser.add_argument("--save_traindata_path", type=str, default="/data/train_valid/", help="Directory for saved train/valid HDF5 data")
    parser.add_argument("--model_dimension", type=int, default=None, help="Grid size. Omit to infer from the first .npz file.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--converge_name", type=str, default="jcp_syn_euler1_loss_convergence", help="Name for convergence plot")
    parser.add_argument("--batch_size", type=int, default=320, help="Per-rank batch size under DDP")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--mode1", type=int, default=24, help="Number of Fourier modes in x")
    parser.add_argument("--mode2", type=int, default=24, help="Number of Fourier modes in y")
    parser.add_argument("--width", type=int, default=16, help="FNO hidden width")
    parser.add_argument("--activation_func", type=str, default="tanh", choices=["tanh", "relu", "sig"], help="Hidden activation function")
    parser.add_argument("--loss_func", type=str, default="mean_square_root", choices=["mean_square_root", "rmse", "root_mse", "root_mean_square", "root_mean_squared", "root_mean_square_error", "sqrt_mse", "msr", "circular_mse", "circular", "angle", "angular", "L2"], help="Training loss: mean_square_root/rmse uses ordinary root-mean-square error in normalized units; circular_mse/circular uses wrapped angular error; L2 is relative L2")
    parser.add_argument("--step_known", type=int, default=1, help="Number of known input timesteps")
    parser.add_argument("--step_predict", type=int, default=1, help="Number of timesteps to predict")
    parser.add_argument("--step_size", type=int, default=499, help="Target offset from the first known frame")
    parser.add_argument("--split_seed", type=int, default=12345, help="Random seed for file-level split and DDP shuffling")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count per rank")
    parser.add_argument("--npz_extract_dir", type=str, default=None, help="Directory used by rank 0 to extract each .npz arr_0.npy before memory-mapped streaming.")
    parser.add_argument("--reuse_existing_h5", action="store_true", help="Reuse existing train/valid/test HDF5 split files instead of regenerating them.")
    parser.add_argument("--ddp_backend", type=str, default=None, choices=[None, "gloo", "nccl"], help="Defaults to nccl on CUDA and gloo on CPU.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    main(
        args.data_path,
        args.save_path,
        args.save_testdata_path,
        args.save_traindata_path,
        args.model_dimension,
        args.epochs,
        args.converge_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mode1=args.mode1,
        mode2=args.mode2,
        width=args.width,
        activation_func=args.activation_func,
        loss_func=args.loss_func,
        step_known=args.step_known,
        step_predict=args.step_predict,
        step_size=args.step_size,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        ddp_backend=args.ddp_backend,
        npz_extract_dir=args.npz_extract_dir,
        reuse_existing_h5=args.reuse_existing_h5,
    )
