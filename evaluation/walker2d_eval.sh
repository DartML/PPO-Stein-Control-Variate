#!/bin/bash

for ((s=13; s<=43; s+=30))
    do
    	for ((i=10; i<=100; i+=10))
            do
		k=$((5000*$i))
		echo $k
		# load Stein PPO and compute variance and save
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 50 -s $s -po FitQ & 
		sleep 1.5s
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 50 -s $s -po MinVar &
		sleep 1.5s

		# load PPO and compute variance and save
		python run.py Walker2d-v1 -ps large -c 0 -p 0 -n $i -sha load -m 50 -s $s &
		sleep 1.5s
    echo $i  
            done
    done 

