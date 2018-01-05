#!/bin/bash

# save a PPO policy
# python run.py Walker2d-v1 -c 0.0 -p 0 -sha save -s 13

# eval on a saved policy
for ((s=13; s<=33; s+=30)) # evaluate on one seed to save time
    do
    	for ((i=10; i<=80; i+=10)) # few number of trajectories to save time
            do
		# load Stein PPO and compute variance and save
		
		# FitQ
		j=$((50*$i))
		echo $j
		python run.py Walker2d-v1 -ps large -p $j -c 1 -n $i -sha load -m 500 -s $s -po FitQ & 
		sleep 1.5s

		# MinVar
		k=$((50*$i))
		echo $k
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 500 -s $s -po MinVar &
		sleep 1.5s
            done
    done 