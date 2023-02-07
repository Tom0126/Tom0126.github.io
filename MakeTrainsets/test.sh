#!/bin/bash

array=(100 20 40,60,80,120,30,50,70,90)
for(( i=0;i<${#array[@]};i++))
  do
      integer=${array[i]}
      echo $integer

  done;
