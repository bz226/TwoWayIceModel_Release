#!/bin/bash
# Control script for polyphase FFT+GBM+Recovery, Florian Steinbach
# Version 160829_1246

# # INFO # # 
# Order of processes performed:
# 1. FFT (elle2fft PPC fft2elle)
# 2. Import FFT data to ellefile
# 3. Calculate DDs from Misorientation overwriting the FFT ones (if switched on)
# 4. Elle process with several topology checks (FS_topocheck)
# 5. "J" Subloop steps with an user defined times of the ReX processes:
#       5a. Recovery
#       5b. Nucleation and Polygonisation -> Topology check after every step 
#       5c. GBM -> Topology check after every step (included in the GBM code)
# 6. FS_flynn2unode_attribute for VISCOSITY
# 7. Create ellefile only for plotting (if switched on) where unodes of one 
#    phase (e.g. air) can be deleted (so this phase is black in plotting) or all 
#    coords are scaled down/up 
# 8. Store all data in output folder and compile zip file

DEBUG=
#DEBUG="echo"
rm *~

#############################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                   INPUT                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#############################################################################

INROOT=fine_foam
OUTPATH="results/"  # output folder, folder "stepXXX" will be stored in here, with / in the end 
OUTROOT=fine_foam
SAVEROOT="results/" # with / in the end
TOTALSTEPS=10       # Number of steps to perform, timestep = 1.5 days, total time = 2 months
STARTSTEP=0         # Steps already performed in starting elle file

SAVESTEPS=1         # Save only every "SAVESTEPS"-th step. Type 1 to save every step

DIMENSIONS=128      # Number of unodes in x- and y-direction
AIRPHASE=0          # What is the phase ID of air (if there is no air, you can leave it at the  
                    # former "air value"
AIRPHASEPLOTLAYER=-1 # -1 means no air, otherwise type airphase
EXCLUDEDPHASE=0     # ID of phase being excluded from DD update in fft2elle 
                    # (no exclusion: set to 0)
REPOSITION=0        # Set to 1 if reposition is needed, to 0 if not

DD_MISORI_FS=0      # Using the code FS_getmisoriDD: Set to 1 to use or to 0 not to use it.
                    # ATTENTION: If you use it, you should also use another recovery code based on
                    #   minimisation of stored strain energies determining DDs from misorientation
                    #   as well!
                    # Also the HAGB angle will always be the one that has been set for recovery

# INPUT FOR RECRYSTALLISATION: 
# The recrystallisation steps will be subdivided in subloops of a number of steps per subloop
# for a total number of "TOTALSUBLOOPS" steps: So for 20 steps that makes 4 subloops á 5 steps.
# You have to distribute the number of subloops and the steps per loop on your own, the number
# steps you indicate is only per subloop!!

TOTALSUBLOOPS=10
LOGSCREEN=0         # Leave at zero, only change to 1 if you wish to switch off
                    # randomisation 
GBM_STEPS=2         # PER SUBLOOP: Number of GBM steps (0=no GBM at all)
    GBM_ATTEMPTS=10     # If GBM crashes, try to do this many times before allowing the whole model to crash
REC_STEPS=2         # PER SUBLOOP: Number of recovery steps (0=do no recovery at all)
NUC_STEPS=2         # PER SUBLOOP: Number of complete nucleation steps (0=do nucleation at all)
    ROT_MOB=500     # Give a value for rotation mobility (1/Pa s m²)
                    # ATTENTION: This will be rotation viscosity when using DDs from misorientations
                    #   and the DD energy recovery code, unit is then Pas!
    HAGB_REC=5      # High angle grain boundary angle for recovery
    HAGB_NUC=5      # High angle grain boundary angle for nucleation
    # Type the name of the recovery and GBM code executable you wish to use
    # ATTENTION: Their output elle files need to be named:
    # recoveryXXX.elle and gbm_pp_fftXXX.elle respectively
    RECOVERY_CODE=FS_recovery 
    GBM_CODE=FS_gbm_pp_fft 

SAVEPLOTLAYERS=0    # Set to 1 if you wish to save a plotlayer file without "$AIRPHASE" 
                    # unodes etc. otherwise set to 0
SCALEBOX4PLOT=0.8     # In pure shear models maybe useful to plot unodes without black
                    # stripes: Type scaling exponent (<1 to scale down) if you want to 
                    # use this or type 0 if you do not want to use this.
                    # Hint: 0.8 is a good value for a 2x1 box with 256² unodes

# CALCULATE AND INCLUDE DATA FOR PASSIVE MARKER GRIDS IN ELLEFILE? 
# --> Type 0 if not, type the dimension (or just $DIMENSIONS) if yes
SAVEGRIDDATA=$DIMENSIONS      # Set to 0 to not to use passive markers or type $DIMENSIONS 

# STORE RESULTS FROM all.out FILE (Type 0 for NO, type 1 for yes)
SAVEALLOUT=1   

# Count the number of grain splits that are done by either rotation recrystallisation 
# (split type 1) grain dissection (split type 2). Type 0 if you do NOT want to use this option
COUNT_SPLITS=1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       USER INFO AND PREPARATIONS                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

echo ~~~~~~~~~~~~~~~~ STARTING ~~~~~~~~~~~~~~~~

# zip all input files to save them:
if [ $STARTSTEP -eq 0 ];
then
    $DEBUG zip -r input_files.zip .
fi

$DEBUG cp $INROOT.elle tmp.elle

# If the directory $OUTPATH does not exist: create it:
if [ ! -d "$OUTPATH" ]; then
    $DEBUG mkdir $OUTPATH    
fi

outadd_files=$OUTPATH"additional_files"
if [ ! -d "$outadd_files" ]; then
    $DEBUG mkdir $outadd_files    
fi


# If the directory $OUTPATH/plotlayers does not exist: create it if user wishes to 
# save plotlayers
if [ $SAVEPLOTLAYERS -ne 0 ];
then
    PLOTLAYER_PATH=$OUTPATH"/plotlayers/"    
    if [ ! -d "$PLOTLAYER_PATH" ]; then
        $DEBUG mkdir $PLOTLAYER_PATH
    fi
fi

# If user uses GBM, store the total amount of gbm steps: 
if [ $GBM_STEPS -ne 0 ]; then
    totalgbmsteps=$((STARTSTEP*TOTALSUBLOOPS*GBM_STEPS))
fi

# If count split file doesn't exists, but user wants to count splits by either
# grain dissection or rotation recrystallisation, create the file with header:
COUNTSPLIT_FILE="Track_SplitEvents.txt"
if [ $COUNT_SPLITS -ne 0 ]; then
    if [ -f $COUNTSPLIT_FILE ]; then
        $DEBUG echo "# # # NEW SIMULATION # # #" >> $COUNTSPLIT_FILE
    else
        $DEBUG echo "# # # Counts of grain splits # # #" > $COUNTSPLIT_FILE
        $DEBUG echo "# Column description (6 columns, 1st one always = 1):" >> $COUNTSPLIT_FILE
        $DEBUG echo "# In case split type = 1:" >> $COUNTSPLIT_FILE
        $DEBUG echo "# 1 split_type phase_olddgrain phase_newgrain id_oldgrain id_newgrain" >> $COUNTSPLIT_FILE
        $DEBUG echo "# In case split type = 2:" >> $COUNTSPLIT_FILE
        $DEBUG echo "# 1 split_type phase_mergedgrain1 phase_mergedgrain2 id_mergedgrain1 id_mergedgrain2" >> $COUNTSPLIT_FILE
        $DEBUG echo "# Split type = 1: rotation recrystallisation" >> $COUNTSPLIT_FILE
        $DEBUG echo "# Split type = 2: grain dissection" >> $COUNTSPLIT_FILE
        $DEBUG echo "# # # NEW SIMULATION # # #" >> $COUNTSPLIT_FILE
    fi  
fi

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         LOOP THROUGH STEPS                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

for (( i=1; i<=$TOTALSTEPS; i++ )) 
do	
	step=$(printf "%03d" $(($i+$STARTSTEP)) )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                    PREPARATIONS FOR THIS STEP                             #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    stepdirectory=$OUTPATH"step"$step"/"
    if [ `echo "$step % "$SAVESTEPS | bc` -eq 0 ];
    then
        $DEBUG mkdir $stepdirectory
    fi

    if [ $COUNT_SPLITS -ne 0 ]; then
        $DEBUG echo "#" >> $COUNTSPLIT_FILE
        $DEBUG echo "# STEP "$(($i+$STARTSTEP))" of "$(($TOTALSTEPS+$STARTSTEP)) >> $COUNTSPLIT_FILE
        $DEBUG echo "#" >> $COUNTSPLIT_FILE
    fi

	echo ~~~~~~~~~~~~~~ STEP $(($i+$STARTSTEP))" of "$(($TOTALSTEPS+$STARTSTEP))  ~~~~~~~~~~~~~~~

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                         ELLE2FFT PROCESS                                  #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    printf "\n~~~~~~~~~~~~~~ ELLE2FFT ~~~~~~~~~~~~~~\n\n"
	$DEBUG FS_elle2fft -i tmp.elle -u $DIMENSIONS -n
	$DEBUG mv elle2fft.elle tmp.elle 

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#           FFT ITSELF AND FFT2ELLE (Florian's "FS_fft2elle")                #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
    printf "\n~~~~~~~~~~~~~~~~ FFT ~~~~~~~~~~~~~~~~\n\n"
	$DEBUG FFT_vs$DIMENSIONS
    printf "\n~~~~~~~~~~~~~~ FFT2ELLE ~~~~~~~~~~~~~~\n\n"
	$DEBUG FS_fft2elle -i tmp.elle -u 1 $EXCLUDEDPHASE $SAVEGRIDDATA -n
	$DEBUG mv fft2elle.elle tmp.elle

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                    REPOSITION (only for simple shear)                     #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if [ $REPOSITION -eq 1 ];
    then
        printf "\n~~~~~~~~~~~~~ REPOSITION ~~~~~~~~~~~~~\n\n"
        $DEBUG reposition -i tmp.elle -n
        $DEBUG mv repos.elle tmp.elle    
    fi

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                   IMPORT MORE DATA FROM tex.out FILE                      #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    printf "\n~~~~~~~~~~~ IMPORT FFT DATA ~~~~~~~~~~~\n\n"
	$DEBUG importFFTdata_florian -i tmp.elle -u 4 5 6 7 12 -s 1 -f 1 -n
	$DEBUG mv fft_out.elle tmp.elle

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#      SPECIALITY OF POLYPHASE: UPDATE PHASE ID (U_VISCOSITY) IN UNODES     #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    $DEBUG FS_flynn2unode_attribute -i tmp.elle -u 1 -n
    $DEBUG mv flynn2unode_attribute.elle tmp.elle

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#         CALCULATE DISLOATION DENSITIES FROM MISORIENTATION                #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if [ $DD_MISORI_FS -ne 0 ];
    then
        printf "\n~~~~~~~~~~~~~~ FS_GETMISORIDD ~~~~~~~~~~~~~~\n\n"     
        $DEBUG cp tmp.elle before_FS_getmisori2dd.elle
	    $DEBUG FS_getmisoriDD -i tmp.elle -u $HAGB_REC -n
	    $DEBUG mv FS_getmisoriDD.elle tmp.elle
    fi

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#           Topology checks (also performed after each GBM step later)      #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    printf "\n~~~~~~~~~~~ TOPOLOGY CHECKS ~~~~~~~~~~~\n\n"
	$DEBUG FS_topocheck -i tmp.elle -n
	$DEBUG mv FS_topocheck.elle tmp.elle
    $DEBUG mv LogfileTopoChecks_DeletedFlynns.txt TopoCheckAfterFFT_DeletedFlynns.txt
    $DEBUG mv LogfileTopoChecks_SplitFlynns.txt TopoCheckAfterFFT_SplitFlynns.txt
    if [ `echo "$i % "$SAVESTEPS | bc` -eq 0 ];
    then
        $DEBUG mv TopoCheckAfterFFT* $stepdirectory
    else
        $DEBUG rm TopoCheckAfterFFT*
    fi

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                            RECRYSTALLISATION IN SUBLOOPS                  #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if [ $TOTALSUBLOOPS -ne 0 ];
    then
    #
    $DEBUG cp tmp.elle before_ReX.elle

    for (( j=1; j<=$TOTALSUBLOOPS; j++ )) 
    do  
        ## PREPARE ##
        subloop3digit=$(printf "%03d" $j )
        logfiledir=$stepdirectory"Logfiles_SubLoop"$subloop3digit
        
        if [ `echo "$i % "$SAVESTEPS | bc` -eq 0 ];
        then
            $DEBUG mkdir $logfiledir
        fi

        ## SUBLOOP NUCLEATION ##
        if [ $NUC_STEPS -ne 0 ];
        then
            printf "\n~~~~~~~~~~~~~~~~~ Nucleation Subloop "$subloop3digit" ~~~~~~~~~~~~~~~~~\n\n"
            nucstep3digit=$(printf "%03d" $NUC_STEPS )
            $DEBUG cp tmp.elle before_nucleation.elle        
            $DEBUG subgrain_unodes_alb -i tmp.elle -u $DIMENSIONS $HAGB_NUC $AIRPHASE 1 -s 1 -f 1 -n
            $DEBUG mv unode_grain001.elle tmp.elle
            #$DEBUG mv unode_grain001.elle unode_grain_step$step"_subloop"$subloop3digit".elle"

            for (( k=1; k<=$NUC_STEPS; k++ )) 
            do 
                # Below: 6 should be the min. number of unodes in a nucleated grain -> is this
                # working??
                $DEBUG u2f_sgb_alb -i tmp.elle -u $DIMENSIONS 6 -s 1 -f 1 -n 
                $DEBUG rm u2f_sgb001.elle # file not needed, here we use the other one: u2f.elle    
                $DEBUG mv u2f.elle tmp.elle
                #$DEBUG mv u2f.elle u2f_step$step"_subloop"$subloop3digit"_nucstep"$k".elle"
            done
            $DEBUG FS_topocheck -i tmp.elle -n
            if [ -f FS_topocheck.elle ];
            then
                $DEBUG mv FS_topocheck.elle tmp.elle
            else
                echo "CONTROL SCRIPT ERROR (NUCLEATION): File FS_topocheck.elle is"\
                     "not existing, stopping the simulation at step "$step " subloop "$subloop3digit
                $DEBUG exit # not using break any more to definitely stop the script at this point
            fi 
        
            if [ $DD_MISORI_FS -ne 0 ];
            then
            printf "\n~~~~~~~~~~~~~~ Update DD after nucleation ~~~~~~~~~~~~~~\n\n"     
                #$DEBUG cp tmp.elle before_misori2dd.elle
	            #$DEBUG misori2dd -i tmp.elle -u $DD_MISORI -n
	            #$DEBUG mv misori2dd001.elle tmp.elle
	            $DEBUG FS_getmisoriDD -i tmp.elle -u $HAGB_REC $AIRPHASE -n
	            $DEBUG mv FS_getmisoriDD.elle tmp.elle
            fi 

        fi
        ## END NUCLEATION ##

        ## SUBLOOP GBM ##
        if [ $GBM_STEPS -ne 0 ]; then
            printf "\n~~~~~~~~~~~~~~~~~ GBM Subloop "$subloop3digit" ~~~~~~~~~~~~~~~~~\n\n"
            gbmstep3digit=$(printf "%03d" $((totalgbmsteps+GBM_STEPS)) )
            $DEBUG cp tmp.elle before_gbm.elle   
            
            # As long as there is not the Elle file of the final step: 
            # Do GBM (the amount of steps intended)
            # Attention: In a microstructure that is prone to errors, this
            # might take a while!!
            attempt=0
            while [ ! -f gbm_pp_fft$gbmstep3digit.elle -a $attempt -lt $GBM_ATTEMPTS ]; do
                attempt=$((attempt+1))
                $DEBUG rm Logfile_* PhaseAreaHistory.txt gbm_pp_fft*
                echo "Trying GBM with "$GBM_STEPS" steps - Attempt number: "$attempt

                $DEBUG $GBM_CODE -i tmp.elle -u 0 0 $LOGSCREEN $totalgbmsteps $AIRPHASE\
                       -s $GBM_STEPS -f $GBM_STEPS -n
            done

            # If after GBM_ATTEMPTS-steps the file still does not exsits: Allow the model to crash:
            if [ ! -f gbm_pp_fft$gbmstep3digit.elle ]; then
                echo "ERROR: GBM crashed more than "$GBM_ATTEMPTS" time(s):"\
                            "Simulation terminated"
                echo ""
                $DEBUG exit
            fi

            # If not: Finally GBM worked out without errors: store the final file, update logfiles and update the total number of GBM steps performed for the next inpput:
            $DEBUG mv gbm_pp_fft$gbmstep3digit.elle tmp.elle
            $DEBUG echo "Step "$step", subloop "$subloop3digit": Finished GBM with"\
                        $attempt" attempt(s)" >> GBMAttempts_Info.txt
            if [ `echo "$i % "$SAVESTEPS | bc` -eq 0 ];
            then
                $DEBUG mv Logfile* $logfiledir
                $DEBUG cp UnodeOriChangeGBM.txt $logfiledir 
            fi
            totalgbmsteps=$((totalgbmsteps+GBM_STEPS))
            
            # Additional topology check after GBM:
            $DEBUG FS_topocheck -i tmp.elle -n
            if [ -f FS_topocheck.elle ];
            then
                $DEBUG mv FS_topocheck.elle tmp.elle
            else
                echo "CONTROL SCRIPT ERROR (GBM): File FS_topocheck.elle is"\
                     "not existing, stopping the simulation at step "$step " subloop "$subloop3digit
                $DEBUG exit # not using break any more to definitely stop the script at this point
            fi 
        fi
        ## END GBM ##

        ## SUBLOOP RECOVERY ##
        if [ $REC_STEPS -ne 0 ];
        then
            printf "\n~~~~~~~~~~~~~~~~~ Recovery Subloop "$subloop3digit" ~~~~~~~~~~~~~~~~~\n\n"
            $DEBUG cp tmp.elle before_recovery.elle        
            $DEBUG $RECOVERY_CODE -i tmp.elle -u $HAGB_REC $AIRPHASE $LOGSCREEN $ROT_MOB 0 $DD_MISORI_FS\
                    -s $REC_STEPS -f $REC_STEPS -n
            recstep3digit=$(printf "%03d" $REC_STEPS )
            if [ -f recovery$recstep3digit.elle ];
            then
                $DEBUG mv recovery$recstep3digit.elle tmp.elle
            else
                echo "CONTROL SCRIPT ERROR (RECOVERY): File recovery$recstep3digit.elle"\
                     " is not existing,"\
                     "stopping the simulation at step "$step " subloop "$subloop3digit
                $DEBUG exit 
            fi 
        fi
        ## END RECOVERY ##
    
    done # with subloops
    
    if [ `echo "$i % "$SAVESTEPS | bc` -eq 0 ];
    then
	    $DEBUG mv GBMAttempts_Info.txt $stepdirectory"GBMAttempts_Info_"$step.txt  
    else        
	    $DEBUG rm GBMAttempts_Info.txt
    fi

    fi # finish of if [ $TOTALSUBLOOPS -ne 0 ]; 

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#      SPECIALITY OF POLYPHASE: UPDATE PHASE ID (U_VISCOSITY) IN UNODES     #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    $DEBUG FS_flynn2unode_attribute -i tmp.elle -u 1 -n
    $DEBUG mv flynn2unode_attribute.elle tmp.elle	
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#   SPECIALITY OF POLYPHASE: CREATE A PLOTLAYER FILE WITHOUT BUBBLE UNODES  #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    if [ $SAVEPLOTLAYERS -ne 0 ];
    then
        printf "\n~~~~~~~~~ CREATE PLOTTING FILE ~~~~~~~~~\n\n"
        $DEBUG FS_create_plotlayer -i tmp.elle -u $AIRPHASEPLOTLAYER $AIRPHASE $SCALEBOX4PLOT 0 0 -n
        $DEBUG mv with_plotlayer.elle tmp_plotlayer.elle
    fi
        
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#                   FINALLY RENAME AND SAVE THE FILES                       #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # Save final ellefile
    if [ `echo "$i % "$SAVESTEPS | bc` -eq 0 ];
    then
        printf "\n~~~~~~~~~~~ STORE ALL FILES ~~~~~~~~~~~\n\n"
        $DEBUG cp tmp.elle $OUTPATH$OUTROOT"_step"$step".elle"
        printf "\n~~~~~~~~~~~ MOVE FILES TO SCRATCH ~~~~~~~~~~~\n\n"
        $DEBUG mv $OUTPATH$OUTROOT"_step"$step".elle" $SAVEROOT
        # Save plotlayer file without bubble unodes if user switched that option on:
        if [ $SAVEPLOTLAYERS -ne 0 ];
        then
            $DEBUG mv tmp_plotlayer.elle $PLOTLAYER_PATH$OUTROOT"_plotlayer_step"$step".elle"
        fi

        # Save other files
	    $DEBUG cp g0.out $stepdirectory"g0_"$step.out 
	    $DEBUG cp tex.out $stepdirectory"tex"$step.out 
	    $DEBUG cp all.out $stepdirectory"all"$step.out
	    $DEBUG cp conv.out $stepdirectory"conv"$step.out
	    $DEBUG cp err.out $stepdirectory"err"$step.out
	    $DEBUG cp temp.out $stepdirectory"temp"$step.out
	    $DEBUG cp temp-FFT.out $stepdirectory"temp-FFT"$step.out
	    $DEBUG cp unodexyz.out $stepdirectory"unodexyz"$step.out
	    $DEBUG cp unodeang.out $stepdirectory"unodeang"$step.out
        $DEBUG mv UnodeOriChangeGBM.txt $stepdirectory"UnodeOriChangeGBM"$step.txt
	    #$DEBUG cp eunodes.data $stepdirectory"eunodes"$step.data
        
	    $DEBUG mv before_ReX.elle $stepdirectory"before_ReX"$step.elle
	    $DEBUG mv before_gbm.elle $stepdirectory"before_gbm"$step.elle
	    $DEBUG mv before_recovery.elle $stepdirectory"before_recovery"$step.elle   
	    $DEBUG mv before_nucleation.elle $stepdirectory"before_nucleation"$step.elle
	    #$DEBUG mv unode_grain* $stepdirectory
	    #$DEBUG mv u2f* $stepdirectory
	    $DEBUG mv before_misori2dd.elle $stepdirectory"before_misori2dd"$step.elle
        $DEBUG mv before_FS_getmisori2dd.elle $stepdirectory"before_FS_getmisori2dd"$step.elle 
        $DEBUG mv recovery.stats $stepdirectory"recovery"$step.stats
        $DEBUG mv ModelStep* $stepdirectory # The logfile folder

        # uncomment the following five lines to zip the step-results additionally:
        #$DEBUG cd $OUTPATH
        #$DEBUG zip step$step.zip $OUTROOT"_step"$step".elle" step$step/*
        #$DEBUG rm -rf step$step
        #$DEBUG rm $OUTROOT"_step"$step".elle"     
        #$DEBUG cd ..
        
        if [ $GBM_STEPS -ne 0 ]; then
            $DEBUG cp initial_stuff.txt $outadd_files"/"
            $DEBUG mv PhaseAreaHistory.txt $outadd_files"/"
            $DEBUG mv UnodeOriChangeGBM.txt $outadd_files"/"
        fi
        if [ $COUNT_SPLITS -ne 0 ]; then
            $DEBUG cp $COUNTSPLIT_FILE $outadd_files"/" 
        fi

        # Zip the step-directory, remove the folder and only keep the zip file:
        cd $stepdirectory
            if [ $SAVEALLOUT -ne 0 ]; then
                # Create dummy Elle file to be able to call FS_statistics:
                $DEBUG echo 'UNODES' >dummy.elle   
                $DEBUG cp "all"$step.out all.out
                $DEBUG cp ../AllOutData.txt .
                $DEBUG FS_statistics -i dummy.elle -u 1 1 -n
                $DEBUG mv AllOutData.txt ../
                $DEBUG rm dummy.elle all.out     
            fi        
            $DEBUG zip -r "step"$step".zip" .
            # $DEBUG mv "step"$step".zip" "../"
            $DEBUG mv "step"$step".zip" $SAVEROOT
        cd ../..
        $DEBUG rm -rf $stepdirectory

    else
        $DEBUG rm tmp_plotlayer.elle *out before_* recovery.stats ModelStep* PhaseAreaHistory*
    fi
    
done

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               ZIP FILES                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# cd $OUTPATH
#     $DEBUG zip -r "ALL_RESULTS.zip" .
# cd ..

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#             DELETE FILES, THEY HAVE BEEN STORED ELSEWHERE                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

$DEBUG rm *.unodes *.flynn *.stats *.out tmp.elle *.data *.poly
echo " "
echo " "
echo ~~~~~~~~~~~~~~~~ FINISHED ~~~~~~~~~~~~~~~~ 
