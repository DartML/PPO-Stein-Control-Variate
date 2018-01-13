'''
state-only control variate
'''
import tensorflow as tf 
import numpy as np 
import math

class MLP_state_function(object):

    def __init__(self, obs_dim,
            name='phi_nn', 
            hidden_sizes=[100, 100], 
            regular_scale=0., fn_type='relu'):
        
        self.obs_dim = obs_dim
        self.name=name
        self.hidden_sizes=hidden_sizes
        self.fn_type = fn_type
        
        if regular_scale == 0.:
            kernel_regularizer = None
        else:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=regular_scale)
        
        self.kernel_regularizer = kernel_regularizer


    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


    def __call__(self, obs_ph, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            hid1_size = self.hidden_sizes[0] 
            hid2_size = self.hidden_sizes[1] 

            obs_dim = self.obs_dim

            if self.fn_type == 'relu': 
                layer1 = tf.layers.dense(obs_ph, hid1_size, tf.nn.relu)
                layer2 = tf.layers.dense(layer1, hid2_size, tf.nn.relu)                
                layer3 = tf.layers.dense(layer2, 1, tf.nn.relu)
                out = tf.identity(layer3)
            elif self.fn_type == 'tanh':
                layer1 = tf.layers.dense(obs_ph, hid1_size, tf.nn.tanh)
                layer2 = tf.layers.dense(layer1, hid2_size, tf.nn.tanh)                
                layer3 = tf.layers.dense(layer2, 1, tf.nn.tanh)
                out = tf.identity(layer3)

            phi_value = tf.squeeze(out)
            phi_act_g= .0

            return phi_value, phi_act_g

    @property
    def phi_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

