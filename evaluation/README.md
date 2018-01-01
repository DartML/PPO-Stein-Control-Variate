# Evaluations of PPO with/without Stein control variate
This is the code of the evaluation part of [Stein control variate](https://arxiv.org/pdf/1710.11198.pdf). It evaluates different variance reduction methods introduced in the paper.


## Running Examples

Take Walker2d-v1 environment as an example.

Train and generate evaluation data:
```shell
# Train Policy
bash walker2d_train.sh

#Evaluation Policy with or without Stein control variates
bash walker2d_eval.sh
```
NB: different max-timesteps lead to different scale of variance: see [results](./results).

Visualize the variance plot of different optimization Phi function methods:

```python
# plot variance figure
python traj_visualize.py
```

## Notice
Currently the evaluation code is a little messy and not tested thoroughly. We will make it clean and easy for testing soon.
