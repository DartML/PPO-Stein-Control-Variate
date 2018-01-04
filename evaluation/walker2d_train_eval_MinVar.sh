#!/bin/bash

# save a PPO policy
# python run.py Walker2d-v1 -c 0.0 -p 0 -sha save -s 13

# eval on a saved policy
for ((s=13; s<=33; s+=30))
    do
    	for ((i=10; i<=100; i+=10))
            do
		# load Stein PPO and compute variance and save
		# FitQ
		j=$((500*$i))
		echo $j
		python run.py Walker2d-v1 -ps large -p $j -c 1 -n $i -sha load -m 500 -s $s -po FitQ & 
		sleep 1.5s
		# MinVar
		k=$((500*$i))
		echo $k
		python run.py Walker2d-v1 -ps large -p $k -c 1 -n $i -sha load -m 500 -s $s -po MinVar &
		sleep 1.5s

		# load PPO and compute variance and save
		python run.py Walker2d-v1 -ps large -c 0 -p 0 -n $i -sha load -m 500 -s $s &
		sleep 1.5s
    echo $i  
            done
    done 