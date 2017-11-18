#!/bin/bash
# Optimizing Phi by FitQ method
cd ..

for ((s=1; s<=5; s+=1))
    do
    echo $s

    # Walker2d-v1 with different training phi iterations
    python train.py Walker2d-v1 -po FitQ -p 500 -s $s -n 500&
    sleep 1.5s
    python train.py Walker2d-v1 -po FitQ -p 1000 -s $s -n 500&
    sleep 1.5s
    python train.py Walker2d-v1 -c 0 -p 0 -s $s -n 500&
    sleep 1.5s

    # HalfCheetah-v1 with different training phi iterations
    python train.py HalfCheetah-v1 -c 0 -p 0 -s $s -n 500&
    sleep 1.5s
    python train.py HalfCheetah-v1 -po FitQ -p 500 -s $s -n 500&
    sleep 1.5s
    python train.py HalfCheetah-v1 -po FitQ -p 1000 -s $s -n 500&
    sleep 1.5s 
    done
