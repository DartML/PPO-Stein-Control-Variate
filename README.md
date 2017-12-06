# Proximal Policy Optimization With Stein Control Variate

In this work, we propose a control variate method to effectively reduce variance for policy gradient methods motivated by Stein's identity.


This repository contains the code of the Proximal Policy Optimization(PPO) with [Stein control variates](https://arxiv.org/pdf/1710.11198.pdf) for Mujoco environments.

The code is based on the excellent implementation of [PPO](https://github.com/pat-coady/trpo).


## Dependencies

* `Python 3.5`
* [`MuJoCo`](http://www.mujoco.org/)
* `TensorFlow 1.3`
* `Gym` - [Installation instructions](https://gym.openai.com/docs).

## Running Experiments

You can run following commands to reproduce our results:

```Shell
cd optimization

# For MinVar optimization
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 
python train.py Walker2d-v1 -b 10000 -ps large -po MinVar -p 500 
python train.py Hopper-v1 -b 10000 -ps large -po MinVar -p 500 
 
python train.py Ant-v1 -b 10000 -ps small -po MinVar -p 500 
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 
python train.py HumanoidStandup-v1 -b 10000 -ps small -po MinVar -p 500 


# For FitQ optimization
python train.py HalfCheetah-v1 -b 10000 -ps large -po FitQ -p 500 
python train.py Walker2d-v1 -b 10000 -ps large -po FitQ -p 500 
python train.py Hopper-v1 -b 10000 -ps large -po FitQ -p 500 

python train.py Ant-v1 -b 10000 -ps small -po FitQ -p 500 
python train.py Humanoid-v1 -b 10000 -ps small -po FitQ -p 500 
python train.py HumanoidStandup-v1 -b 10000 -ps small -po FitQ -p 500


#For baseline PPO
python train.py HalfCheetah-v1 -b 10000 -ps large -c 0
python train.py Walker2d-v1 -b 10000 -ps large -c 0
python train.py Hopper-v1 -b 10000 -ps large -c 0

python train.py Ant-v1 -b 10000 -ps small -c 0
python train.py Humanoid-v1 -b 10000 -ps small -c 0
python train.py HumanoidStandup-v1 -b 10000 -ps small -c 0
```
The log files is in optimization/dartml_data. Further, we provide two shell scripts for tuning hyperparameters of stein control variates in the [scripts](optimization/scripts) folder.

For evaluation of PPO with/without Stein control variate, please see [here](evaluation).

## Citations
If you find Stein control variates helpful, please cite following papers:

>[Sample-efficient Policy Optimization with Stein Control Variate.](https://arxiv.org/pdf/1710.11198.pdf)
>Hao Liu\*, Yihao Feng\*, Yi Mao, Dengyong Zhou, Jian Peng, Qiang Liu (*: equal contribution).
>Preprint 2017

## Feedbacks

If you have any questions about the code or the paper, please feel free to [contact us](mailto:yihaof95@gmail.com).











