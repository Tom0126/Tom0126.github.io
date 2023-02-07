#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh

inputfile=/lustre/collider/xuzixun/software/siyuancalo/cepc-calo/build/small_hcal_e+_120GeV_2cm_1k.root
outputfile=/lustre/collider/songsiyuan/test.npy
python /home/songsiyuan/CEPC/PID/MakeTrainsets/MakeTrainsets.py --infile $inputfile --outfile $outputfile
#python MakeTrainsets.py
#python Test.py
