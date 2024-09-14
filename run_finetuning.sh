#!/bin/bash

cd fine_tuning

trA='gcc' 
for teA in gcc #gccsnr-20 gccsnr-10 gccsnr0 gccsnr10
do
    python main_ICL_finetuning.py \
        -trA $trA -teA $teA >> nohup_SSL_finetuning.log
done