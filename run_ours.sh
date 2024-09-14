#!/bin/bash

cd ours

trA='gcc' 
for teA in gcc #gccsnr-20 gccsnr-10 gccsnr0 gccsnr10
do
    python main_ICL_ours.py \
        -trA $trA -teA $teA \
        -epoch 30 \
        -incremental \
        -phaseN 10 \
        -rg 1e-1 \
        -Hidden 20000 > nohup_ours_SSL_incremental.log
done