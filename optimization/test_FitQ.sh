#!/bin/bash

for ((s=1; s<=5; s+=1))
    do
    echo $s
    python train.py Walker2d-v1 -po FitQ -p 500 -s $s -n 500&
    sleep 1.5s
    python train.py Walker2d-v1 -po FitQ -p 1000 -s $s -n 500&
    sleep 1.5s
    python train.py Walker2d-v1 -po FitQ -c 0 -p 0 -s $s -n 500&
    sleep 1.5s
    python train.py HalfCheetah-v1 -po FitQ -c 0 -p 0 -s $s -n 500&
    sleep 1.5s
    python train.py HalfCheetah-v1 -po FitQ -p 500 -s $s -n 500&
    sleep 1.5s
    python train.py HalfCheetah-v1 -po FitQ -p 1000 -s $s -n 500&
    sleep 1.5s 
    done

