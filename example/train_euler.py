import sys
sys.path.append('../src/')

import argparse
from timeit import default_timer

import torch
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from model_euler import FNO2d
from utilities3_grain import count_params, LpLoss
from utilities3_grain import smoothness_loss, log_normalize_np, reference_normalize_np, smoothness_loss
from utilities3_grain import mem_needed_bytes, human


def configure_locale():
    """Set a UTF-8 locale for multiprocessing workers if the shell env is incomplete."""
    if not os.environ.get("LANG"):
        os.environ["LANG"] = "en_US.UTF-8"
    if not os.environ.get("LC_CTYPE"):
        os.environ["LC_CTYPE"] = os.environ["LANG"]


def build_dataloader(dataset, batch_size, shuffle, num_workers):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(**loader_kwargs)

class H5Dataset(Dataset):
    """
    PyTorch Dataset that reads samples from an HDF5 file on-the-fly.
    This avoids loading the entire dataset into memory at once.
    """

    def __init__(self, h5_file_path):
        super().__init__()
        self.h5_file_path = h5_file_path
        # open the file once here to get the dataset shapes (and close immediately)
        with h5py.File(self.h5_file_path, 'r') as f:
            # first dimension is number of samples
            # require that all relevant datasets have the same N
            self.N = f['euler_known'].shape[0]
            # check that each dataset has the same shape[0]
            for name in ['euler_predict', 'strain_rate', 'temperature', 'pressure']:
                assert f[name].shape[0] == self.N, f"Inconsistent N for dataset {name}"
            
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as f:
            # grab the arrays for sample index idx
            # shape for each is [128, 128, step_known/predict]
            euler_known   = f['euler_known'][idx]
            euler_predict = f['euler_predict'][idx]
            strain_rate    = f['strain_rate'][idx]
            temperature    = f['temperature'][idx]
            pressure       = f['pressure'][idx]

        euler_known   = torch.from_numpy(euler_known).float()
        euler_predict = torch.from_numpy(euler_predict).float()
        strain_rate    = torch.from_numpy(strain_rate).float()
        temperature    = torch.from_numpy(temperature).float()
        pressure       = torch.from_numpy(pressure).float()

        return euler_known, euler_predict, strain_rate, temperature, pressure

def summarize_hdf5_dataset(h5_file_path, dataset_name, chunk_size=1024):
    """
    Prints shape, dtype, and min/max range of the dataset in `h5_file_path`
    Loads data in chunks to avoid large memory usage
    """
    with h5py.File(h5_file_path, 'r') as f:
        dset = f[dataset_name]
        shape = dset.shape
        dtype = dset.dtype
        print(f"{dataset_name} shape: {shape}, dtype: {dtype}")
        global_min = float('inf')
        global_max = float('-inf')
        N = int(dset.shape[0])
        indices = np.random.permutation(N)# first dimension is sample dimension
        # Read in small chunks along axis 0
        for i,index in enumerate(indices):
            if (i<N/100):
                # This slice is shape [end-start, 256, step]
                chunk_data = dset[index,:,0]
                # get min/max for this chunk
                local_min = chunk_data.min()
                local_max = chunk_data.max()
                if local_min < global_min:
                    global_min = local_min
                if local_max > global_max:
                    global_max = local_max
        print(f"  data range for random {int(N/100)} samples: [{global_min}, {global_max}]\n")

def save_train_data_h5(save_file_name,grid_size,step_known,step_predict,train_index,euler_known,euler_predict,strain_rate,temperature,pressure):
    with h5py.File(save_file_name, 'w') as f:
        # create datasets with final shapes
        euler_known_train_dset = f.create_dataset(
            'euler_known', 
            shape=(len(train_index), grid_size, grid_size,step_known),
            dtype=euler_known.dtype
        )
        euler_predict_train_dset = f.create_dataset(
            'euler_predict', 
            shape=(len(train_index), grid_size, grid_size, step_predict),
            dtype=euler_predict.dtype
        )
        strain_rate_train_dset = f.create_dataset(
            'strain_rate', 
            shape=(len(train_index), grid_size, grid_size, step_known),
            dtype=strain_rate.dtype
        )
        temperature_train_dset = f.create_dataset(
            'temperature', 
            shape=(len(train_index), grid_size, grid_size, step_known),
            dtype=temperature.dtype
        )
        pressure_train_dset = f.create_dataset(
            'pressure', 
            shape=(len(train_index), grid_size, grid_size, step_known),
            dtype=pressure.dtype
        )
        # 2) copy in small batches to avoid big memory usage
        chunk_size = 1024  # adjust this to fit in RAM
        for start in range(0, len(train_index), chunk_size):
            end = start + chunk_size
            print(f'done transferring {end} samples')
            batch_idx = train_index[start:end]
            # each of these advanced-index calls makes a copy of the slice
            # but only of size [chunk_size, ...]
            euler_known_train_dset[start:end]   = euler_known[batch_idx, :, :, :]
            euler_predict_train_dset[start:end] = euler_predict[batch_idx, :, :, :]
            strain_rate_train_dset[start:end]    = strain_rate[batch_idx, :, :, :]
            temperature_train_dset[start:end]    = temperature[batch_idx, :, :, :]
            pressure_train_dset[start:end]       = pressure[batch_idx, :, :, :]
    print(f"Saved train data to {save_file_name}")


def load_data(which_euler,data_path,save_traindata_path,H_ref_bound,S_ref_bound,
              grid_size=1000,step_known=1, step_predict=1, step_size = 24,
              batch_size = 2, shuffle=True, num_workers=0):
    
    """ Load the data from the given paths and split into train, validation and test sets

    Parameters
    ----------
    step_known: how many steps are known
    step_predict: how many steps are predicted
    step_size: steps in between
    * * *     - - - - - - - - -     * * *
    ↑___↑    ↑________________↑     ↑___↑ 
    step         step size           step
    known                            pred
    """

    # load the input data
    print(data_path)
    files = os.listdir(data_path)
    # filter for .npz files
    if which_euler == 'euler1':
        euler_files = [x for x in files if x.endswith(".npz") and (x.startswith("euler_1"))]
    elif which_euler == 'euler2':
        euler_files = [x for x in files if x.endswith(".npz") and (x.startswith("euler_2"))]
    elif which_euler == 'euler3':
        euler_files = [x for x in files if x.endswith(".npz") and (x.startswith("euler_3"))]
    else:
        raise ValueError(f"Missing an arg input which_euler!!")
    # count the npz files
    euler_npz_count = len(euler_files)
    print("-----------------------------------\n-----------------------------------")
    print(f"Number of npz files to read: {euler_npz_count}")
    print("-----------------------------------")
    if (euler_npz_count==0):
        raise ValueError(f"Zero data file!! Check your path to your data folder!!")
    # 1) gather all n's first
    all_n = []
    for euler_file in euler_files:
        # load just enough to compute n, or parse file shape
        euler_path = os.path.join(data_path, euler_file)
        shape = np.load(euler_path)['arr_0'].shape 
        # compute n from shape[-1]
        n = shape[-1] - step_size - step_known - step_predict + 1
        all_n.append(n)
    total_samples = sum(all_n)
    # 2) pre-allocate big arrays
    euler_known    = np.empty((total_samples, grid_size, grid_size, step_known), dtype=np.float32)
    euler_predict  = np.empty((total_samples, grid_size, grid_size, step_predict), dtype=np.float32)
    strain_rate    = np.empty((total_samples, grid_size, grid_size, step_known), dtype=np.float32)
    temperature    = np.empty((total_samples, grid_size, grid_size, step_known), dtype=np.float32)
    pressure       = np.empty((total_samples, grid_size, grid_size, step_known), dtype=np.float32)
    print(f'\ntotal # of training samples: {total_samples}')
    bytes_needed = mem_needed_bytes(total_samples, grid_size, step_known, step_predict, dtype_bytes=4)  # float32
    print(f'est. storage for each input/output: {human(bytes_needed)}\n')
    
    # 3) loop again to fill in data slices
    offset = 0
    for i, euler_file in enumerate(euler_files):
        euler_path = os.path.join(data_path, euler_file)
        euler_data = np.load(euler_path)['arr_0']
        print(f"Read file: {euler_file}, shape: {euler_data.shape}, num samples: {all_n[i]}, value range: ",euler_data[0,:,:,:].max(),euler_data[0,:,:,:].min())
        n_i = all_n[i]
        # create these small arrays or slice directly
        for j in range(n_i):
            # --- euler Known ---
            # Divide by 180 for euler 1, 3. /45 for euler 2
            known_slice = euler_data[0, :, :, j : j + step_known].astype(np.float32)
            if which_euler == 'euler2':
                known_slice = (known_slice - 45.0) /45.0 
            else:
                known_slice = (known_slice) /180.0 
            euler_known[offset + j] = known_slice
            # --- euler Predict ---
            predict_slice = euler_data[0, :, :, (j + step_size + step_known)
                                        : (j + step_size + step_known + step_predict)]
            predict_slice = predict_slice.astype(np.float32)
            if which_euler == 'euler2':
                predict_slice = (predict_slice - 45.0) /45.0 
            else:
                predict_slice = (predict_slice) /180.0 
            euler_predict[offset + j] = predict_slice
            # --- Strain Rate (log normalize) ---
            sr_slice = euler_data[1, :, :, j : (j + step_known)].astype(np.float32)
            sr_slice = log_normalize_np(sr_slice, min_ref=S_ref_bound[0], max_ref=S_ref_bound[1])
            strain_rate[offset + j] = sr_slice
            # --- Pressure + offset + reference normalize ---
            pr_slice = euler_data[2, :, :, j : (j + step_known)].astype(np.float32)
            # pr_slice += 900.0 * 9.80665  # add offset
            pr_slice = reference_normalize_np(pr_slice, min_ref=H_ref_bound[0], max_ref=H_ref_bound[1])
            pressure[offset + j] = pr_slice
            # --- Temperature: invert it, 1/T ---
            temp_slice = euler_data[3, :, :, j : (j + step_known)].astype(np.float32)
            # watch out for zeros or very small T that cause inf
            # temp_slice = np.where(temp_slice == 0, np.finfo(np.float32).tiny, temp_slice)
            temp_slice = 1.0 / temp_slice
            # temp_slice = reference_normalize_np(temp_slice, min_ref=T_ref_bound[0], max_ref=T_ref_bound[1])
            temperature[offset + j] = temp_slice
        offset += n_i

    print(f"-------- Done reading all data from {euler_npz_count} files ---------")
    print("------------------------------------------------------")
    
    # stop if the normalized data is outside of [-1,1]
    epsilon = 1e-5
    if(np.min(euler_known)<-1-epsilon or np.max(euler_known)>1+epsilon):
        raise ValueError(f"euler_known is outside [-1,1]. min = {np.min(euler_known)} max = {np.max(euler_known)}.Check reference min and max used to normalize the data.")
    elif(np.min(euler_predict)<-1-epsilon or np.max(euler_predict)>1+epsilon):
        raise ValueError(f"euler_predict is outside [-1,1]. min = {np.min(euler_predict)} max = {np.max(euler_predict)}.Check reference min and max used to normalize the data.")
    elif(np.min(strain_rate)<-1-epsilon or np.max(strain_rate)>1+epsilon):
        raise ValueError(f"strainrate is outside [-1,1]. min = {np.min(strain_rate)} max = {np.max(strain_rate)}.Check reference min and max used to normalize the data.")
    elif(np.min(pressure)<-1-epsilon or np.max(pressure)>1+epsilon):
        raise ValueError(f"pressure is outside [-1,1]. min = {np.min(pressure)} max = {np.max(pressure)}.Check reference min and max used to normalize the data.")
    elif(np.min(temperature)<-1-epsilon or np.max(temperature)>1+epsilon):
        raise ValueError(f"temperature is outside [-1,1]. min = {np.min(temperature)} max = {np.max(temperature)}.Check reference min and max used to normalize the data.")
    
    # split the data into training, validation and test sets in the ratio 80:10:10 randomly
    random_index = np.random.permutation(strain_rate.shape[0])
    train_index = random_index[0:int(0.8 * strain_rate.shape[0])]
    valid_index = random_index[int(0.8 * strain_rate.shape[0]):int(0.9 * strain_rate.shape[0])]
    test_index  = random_index[int(0.9 * strain_rate.shape[0]):]

    # train data
    euler_known_train   = euler_known[train_index, :, :, :]
    euler_predict_train = euler_predict[train_index, :, :, :]
    strain_rate_train  = strain_rate[train_index, :, :, :]
    temperature_train = temperature[train_index, :, :, :]
    pressure_train = pressure[train_index, :, :, :]

    # validation data
    euler_known_valid = euler_known[valid_index, :, :, :]
    euler_predict_valid = euler_predict[valid_index, :, :, :]
    strain_rate_valid = strain_rate[valid_index, :, :, :]
    temperature_valid = temperature[valid_index, :, :, :]
    pressure_valid = pressure[valid_index, :, :, :]

    # test data
    euler_known_test = euler_known[test_index, :, :, :]
    euler_predict_test = euler_predict[test_index, :, :, :]
    strain_rate_test = strain_rate[test_index, :, :, :]
    temperature_test = temperature[test_index, :, :, :]
    pressure_test = pressure[test_index, :, :, :]
    
    print("-----------------------------------------------------------------")
    print("Size of the data       :")
    print(f"{'Shape of euler_known_train'       :<30} : {euler_known_train.shape}   data range: [{euler_known_train.min():.2f}, {euler_known_train.max():.2f}]")
    print(f"{'Shape of euler_predict_train'     :<30} : {euler_predict_train.shape}   data range: [{euler_predict_train.min():.2f}, {euler_predict_train.max():.2f}]")
    print(f"{'Shape of strain_rate_train'       :<30} : {strain_rate_train.shape}   data range: [{strain_rate_train.min():.2f}, {strain_rate_train.max():.2f}]")
    print(f"{'Shape of temperature_train'       :<30} : {temperature_train.shape}   data range: [{temperature_train.min():.2f}, {temperature_train.max():.2f}]")
    print(f"{'Shape of pressure_train'          :<30} : {pressure_train.shape}   data range: [{pressure_train.min():.2f}, {pressure_train.max():.2f}]")
    print(f"{'Shape of euler_known_valid'       :<30} : {euler_known_valid.shape}   data range: [{euler_known_valid.min():.2f}, {euler_known_valid.max():.2f}]")
    print(f"{'Shape of euler_predict_valid'     :<30} : {euler_predict_valid.shape}   data range: [{euler_predict_valid.min():.2f}, {euler_predict_valid.max():.2f}]")
    print(f"{'Shape of strain_rate_valid'       :<30} : {strain_rate_valid.shape}   data range: [{strain_rate_valid.min():.2f}, {strain_rate_valid.max():.2f}]")
    print(f"{'Shape of temperature_valid'       :<30} : {temperature_valid.shape}   data range: [{temperature_valid.min():.2f}, {temperature_valid.max():.2f}]")
    print(f"{'Shape of pressure_valid'          :<30} : {pressure_valid.shape}   data range: [{pressure_valid.min():.2f}, {pressure_valid.max():.2f}]")
    print(f"{'Shape of euler_known_test'        :<30} : {euler_known_test.shape}   data range: [{euler_known_test.min():.2f}, {euler_known_test.max():.2f}]")
    print(f"{'Shape of euler_predict_test'      :<30} : {euler_predict_test.shape}   data range: [{euler_predict_test.min():.2f}, {euler_predict_test.max():.2f}]")
    print(f"{'Shape of strain_rate_test'        :<30} : {strain_rate_test.shape}   data range: [{strain_rate_test.min():.2f}, {strain_rate_test.max():.2f}]")
    print(f"{'Shape of temperature_test'        :<30} : {temperature_test.shape}   data range: [{temperature_test.min():.2f}, {temperature_test.max():.2f}]")
    print(f"{'Shape of pressure_test'           :<30} : {pressure_test.shape}   data range: [{pressure_test.min():.2f}, {pressure_test.max():.2f}]")
    print("-----------------------------------------------------------------")

    # # save test data
    save_train_data_h5(save_traindata_path+'euler_test_data.h5',
                       grid_size,step_known,step_predict,test_index,
                       euler_known,euler_predict,strain_rate,temperature,pressure)

    # # save train data
    save_train_data_h5(save_traindata_path+'euler_train_data.h5',
                       grid_size,step_known,step_predict,train_index,
                       euler_known,euler_predict,strain_rate,temperature,pressure)

    # save valid data
    save_train_data_h5(save_traindata_path+'euler_valid_data.h5',
                       grid_size,step_known,step_predict,valid_index,
                       euler_known,euler_predict,strain_rate,temperature,pressure)
    
    # test if data can be read correctly
    train_dataset = H5Dataset(save_traindata_path+'euler_train_data.h5')
    valid_dataset = H5Dataset(save_traindata_path+'euler_valid_data.h5')
    for name in ["euler_known", "euler_predict", "strain_rate", "temperature", "pressure"]:
        summarize_hdf5_dataset(save_traindata_path+'euler_train_data.h5', name, chunk_size=512)
    for name in ["euler_known", "euler_predict", "strain_rate", "temperature", "pressure"]:
        summarize_hdf5_dataset(save_traindata_path+'euler_valid_data.h5', name, chunk_size=512)
    
    train_loader = build_dataloader(train_dataset, batch_size, shuffle, num_workers)
    valid_loader = build_dataloader(valid_dataset, batch_size, shuffle, num_workers)
    return train_loader, valid_loader, len(euler_files)


def train(model, train_loader, valid_loader, optimizer, 
          myloss, epoch, device, T_in, T_end, step, batch_size,smoothness_weight,GRAD_CLIP):

    print(f"\nThe model has {count_params(model)} trainable parameters\n")
    print(f"Training the model on {device} for {epoch} epoch ...\n")

    train_loss = torch.zeros(epoch)
    test_loss  = torch.zeros(epoch)
    
    for ep in range(epoch):
        model.train()
        t1 = default_timer()
        train_l2_step_training = 0
        test_l2_step_testing = 0
        # xx is input, yy is prediction
        for xx, yy, C1, C2, C3 in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            C1 = C1.to(device)
            C2 = C2.to(device)
            C3 = C3.to(device)

            for t in range(0, (T_end - T_in),step):
                # read in the time step to predict
                y = yy[..., t:t + step]
                im_train = model(xx,C1,C2,C3) 
                smooth_penalty = smoothness_loss(im_train[:,:,:,0])
                loss +=  myloss(im_train.reshape(batch_size, -1), y.reshape(batch_size, -1))
                loss += smoothness_weight * smooth_penalty
                smooth_loss_ratio = smoothness_weight * smooth_penalty/loss
                xx = torch.cat((xx[..., step:], im_train), dim=-1)

                # Check predictions
                if torch.isnan(im_train).any() or torch.isinf(im_train).any():
                    print(f"NaN/Inf in predictions at epoch {epoch}, step {t}")
                # end of one time slice prediction
                # -------
            
            # add the loss of this batch to all batches
            train_l2_step_training += loss.item()
            train_loss[ep] = train_l2_step_training
            # clear any previously accumulated gradients (from the previous batch)
            optimizer.zero_grad()
            # backpropagate the loss for this batch
            loss.backward()
            # clip gradients if their total norm exceeds 5
            if GRAD_CLIP:
                total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                total_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
            # update the model parameters using the gradients
            optimizer.step()
        del xx, yy, C1, C2, C3 
        # end of each training batch

        with torch.no_grad():
            for xx, yy, C1, C2, C3 in valid_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                C1 = C1.to(device)
                C2 = C2.to(device)
                C3 = C3.to(device)

                for t in range(0, (T_end - T_in),step):
                    y = yy[..., t:t + step]
                    im_test = model(xx,C1,C2,C3)
                    smooth_penalty = smoothness_loss(im_test[:,:,:,0])
                    loss += myloss(im_test.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    loss += smoothness_weight * smooth_penalty

                    xx = torch.cat((xx[..., step:], im_test), dim=-1)

                test_l2_step_testing += loss.item()
                test_loss[ep] = test_l2_step_testing

        t2 = default_timer()
        if GRAD_CLIP:
            print(f"current epoch {ep}, time {(t2-t1)/60:.1f} min, train loss {train_l2_step_training:.3f} ({smooth_loss_ratio*100:.0f}% smooth loss), valid loss: {test_l2_step_testing:.3f}, max norm before: {total_norm_before:.3f}, after: {total_norm_after:.3f}, learning rate: {optimizer.param_groups[0]['lr']}")
        else:
            print(f"current epoch {ep}, time {(t2-t1)/60:.1f} min, train loss {train_l2_step_training:.3f} ({smooth_loss_ratio*100:.0f}% smooth loss), valid loss: {test_l2_step_testing:.3f}, max norm: {torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)):.3f}, learning rate: {optimizer.param_groups[0]['lr']}")

    return train_loss, test_loss


def main(which_euler,data_path, save_path,save_traindata_path,model_dimension, epochs_input, num_workers):
    """ Train the model using the given data
    """
    configure_locale()

    # use clip only if training loss is wild
    GRAD_CLIP = False
    BATCH_AVERAGE = True

    # define the hyperparameters
    # may need fine tuning according to each training data set
    learning_rate   = 0.0001
    smoothness_weight = 0.0
    batch_size      = 30
    mode1           = 64 # 64
    mode2           = 64
    width           = 64 # 32
    activation_func = 'tanh' #'tanh'
    loss_func       = 'L2'

    epochs = epochs_input
    grid_size = model_dimension

    # define num steps to predict
    step_known = 1       # num of steps whose info is known, i.e. step 1-3 are given
    step_predict = 1    # num of steps whose info is to perdict, i.e. step 4-10 are to be predict given 
    step_size = 0

    # define non-dimensionalization reference, first is low values, seccond is high value
    T_ref_bound = np.array([-26.0,-10.20])
    H_ref_bound = np.array([1*900*9.80665, 1000*900*9.80665])
    kde_ref_bound = np.array([0.0, 0.183])
    S_ref_bound = np.array([21e-14, 1800001e-14])

    # define the device for training (only on one GPU if available)
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DataLoader num_workers: {num_workers}")
    
    
    train_loader, valid_loader, num_npz_files = load_data(which_euler,data_path,save_traindata_path,H_ref_bound,S_ref_bound,
                                                      grid_size, step_known, step_predict, 
                                                      step_size,batch_size,True,num_workers)
    
    # define the model
    print("-----------------------------------------------------------------")
    print("\n Training using activation function: ",activation_func, ", loss function: ",loss_func, ", loss average over batch size?: ",BATCH_AVERAGE)
    print("\n Recurring property: steps known",step_known, ", steps to predict: ",step_predict)
    print(f"  Model parameters: mode1: {mode1}, mode2: {mode2}, width: {width}, batch size: {batch_size}, base learning rate: {learning_rate}")
    model = FNO2d(mode1, mode2, width,step_known,activation_func,loss_func).to(device)
    # print(model,'\n',model.parameters())
    
    # define the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # define the loss function
    myloss = LpLoss(size_average=BATCH_AVERAGE)
    
    # train the model
    train_loss, valid_loss = train(model, train_loader, valid_loader, 
                                  optimizer,  myloss, 
                                  epochs, device, step_known, step_known+step_predict, 
                                  1, batch_size,smoothness_weight,GRAD_CLIP)
    
    # save the model and the loss tensors
    torch.save(model.state_dict(), save_path+'_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'.pth') #
    torch.save(train_loss, save_path + '_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'_train_loss.pt')
    torch.save(valid_loss, save_path + '_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'_valid_loss.pt') 
    print("Model saved to: ",save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str, default='./../data/', help='Path to data')
    parser.add_argument('--save_path', type=str, default='./../model/model', help='Path to saved model')
    parser.add_argument('--save_traindata_path', type=str, default='./../data/', help='Path to saved train data')
    parser.add_argument('--model_dimension',type=int,default=128,help='Number of grid points in x or y. Currently only support square matrix.')
    parser.add_argument('--epochs',type=int,default=20,help='Num of epochs')
    parser.add_argument('--which_euler',type=str,default='euler1',help='Which euler angle to train.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes')
    args = parser.parse_args()

    main(args.which_euler,args.data_path,args.save_path,args.save_traindata_path,args.model_dimension,args.epochs,args.num_workers)
