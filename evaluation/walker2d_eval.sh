#!/bin/bash

for ((s=13; s<=43; s+=30))
    do
    	for ((i=5; i<=50; i+=5))
            do
		k=$((2000*$i))
		echo $k
		# load Stein PPO and compute variance and save
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 500 -s $s -po FitQ
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 500 -s $s -po MinVar

		# load PPO and compute variance and save
		python run.py Walker2d-v1 -ps large -c 0 -p 0 -n $i -sha load -m 500 -s $s
    echo $i  
            done
    done 

