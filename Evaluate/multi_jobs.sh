#!/bin/bash
#e,pi
#array=(100 20 40 60 80 120 30 50 70 90)
#mu
array=(160)
for(( i=0;i<${#array[@]};i++))
do

    cp multi_submit.sh scripts/multi_submit${array[i]}.sh
    #e
    #sed -i 7c/lustre/collider/songsiyuan/CEPC/PID/e+/resultFile/hcal_e+_${array[i]}GeV_2cm_10k.root scripts/multi_submit${array[i]}.sh
    #sed -i 8c/lustre/collider/songsiyuan/CEPC/PID/Trainsets/hcal_e+_${array[i]}GeV_2cm_10k.npy scripts/multi_submit${array[i]}.sh
    sed -i 7cinputfile\=/lustre/collider/songsiyuan/CEPC/PID/mu+/hcal_muon_ ${array[i]}GeV_2cm_10w.root scripts/multi_submit${array[i]}.sh
    sed -i 8coutputfile\=/lustre/collider/songsiyuan/CEPC/PID/Trainsets/hcal_muon_${array[i]}GeV_2cm_10w.npy scripts/multi_submit${array[i]}.sh
    sed -i 2cExecutable\=./scripts/multi_submit${array[i]}.sh multi_python.sub
    condor_submit multi_python.sub

done;
