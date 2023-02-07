#!/bin/bash
#e,pi
array=(100 20 40 60 80 120 30 50 70 90)

for(( i=0;i<${#array[@]};i++))
do

    cp multi_submit.sh scripts/multi_submit_pi_${array[i]}.sh
    cp multi_python.sub scripts/multi_python_pi_${array[i]}.sub
    #e
    sed -i 7cinputfile\=/lustre/collider/songsiyuan/CEPC/PID/pion+/resultFile/hcal_pi+_${array[i]}GeV_2cm_10k.root scripts/multi_submit_pi_${array[i]}.sh
    sed -i 8coutputfile\=/lustre/collider/songsiyuan/CEPC/PID/Trainsets/hcal_pi+_${array[i]}GeV_2cm_10k.npy scripts/multi_submit_pi_${array[i]}.sh
    sed -i 2cExecutable\=./multi_submit_pi_${array[i]}.sh scripts/multi_python_pi_${array[i]}.sub
    condor_submit scripts/multi_python_pi_${array[i]}.sub

done;