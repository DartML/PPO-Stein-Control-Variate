'''
Stein control variate
'''
import tensorflow as tf 
import numpy as np 
import math

class ContinousMLPPhiFunction(object):

    def __init__(self, obs_dim, 
            act_dim, name='phi_nn', 
            hidden_sizes=[100, 100], 
            regular_scale=0., fn_type='relu'):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
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


    def __call__(self, obs_ph, act_ph, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            hid1_size = self.hidden_sizes[0] 
            hid2_size = self.hidden_sizes[1] 

            obs_dim = self.obs_dim
            act_dim = self.act_dim

            W1 = self.variable([obs_dim,hid1_size],obs_dim)
            b1 = self.variable([hid1_size],obs_dim)
            W2 = self.variable([hid1_size,hid2_size],hid1_size+act_dim)
            W2_action = self.variable([act_dim,hid2_size],hid1_size+act_dim)
            b2 = self.variable([hid2_size],hid1_size+act_dim)
            W3 = tf.Variable(tf.random_uniform([hid2_size,1],-3e-3,3e-3))
            b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

            if self.fn_type == 'relu': 
                layer1 = tf.nn.relu(tf.matmul(obs_ph, W1) + b1)
                layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(act_ph, W2_action) + b2)
                out = tf.identity(tf.matmul(layer2,W3) + b3)
            elif self.fn_type == 'tanh':
                layer1 = tf.tanh(tf.matmul(obs_ph, W1) + b1)
                layer2 = tf.tanh(tf.matmul(layer1,W2) + tf.matmul(act_ph, W2_action) + b2)
                out = tf.identity(tf.matmul(layer2,W3) + b3)

            phi_value = tf.squeeze(out)
            phi_act_g= tf.gradients(phi_value, act_ph)[0]

            return phi_value, phi_act_g

    @property
    def phi_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

