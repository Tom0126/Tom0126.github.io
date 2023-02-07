1. Root file -> .npy file
    cd scripts
    change parameters in *_jobs.sh
    ./e_jobs.sh,mu_jobs.sh,pi_jobs.sh

2. Combine distinct .npy files into one file
    change parameters in Combine.py
    condor_submit combine.sub

3. Split combined ,npy files
    change parameters in Split.py
    condor_submit split.sub

4. make final datasets # do step 4, step 3, 4 could be neglected

    change parameters in MakeFinalDatasets.py
    condor_submit make_final_dataset.sub


