#!/bin/bash

#mu
array=(160)
for(( i=0;i<${#array[@]};i++))
do

    cp multi_submit.sh scripts/multi_submit_mu_${array[i]}.sh
    cp multi_python.sub scripts/multi_python_mu_${array[i]}.sub

    sed -i 7cinputfile\=/lustre/collider/songsiyuan/CEPC/PID/mu+/hcal_muon_${array[i]}GeV_2cm_10w.root scripts/multi_submit_mu_${array[i]}.sh
    sed -i 8coutputfile\=/lustre/collider/songsiyuan/CEPC/PID/Trainsets/hcal_muon_${array[i]}GeV_2cm_10w.npy scripts/multi_submit_mu_${array[i]}.sh
    sed -i 2cExecutable\=./multi_submit_mu_${array[i]}.sh scripts/multi_python_mu_${array[i]}.sub
    condor_submit scripts/multi_python_mu_${array[i]}.sub

done;
