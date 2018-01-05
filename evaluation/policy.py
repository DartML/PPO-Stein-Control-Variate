"""
    Policy Optimization Policy with Stein control variates
"""

import os
import pickle
import numpy as np
import tensorflow as tf

import tb_logger as logger
from phi_functions.ContinousMLPPhiFunction import ContinousMLPPhiFunction


class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, 
                obs_dim, 
                act_dim, 
                kl_targ, 
                epochs, 
                phi_epochs, 
                policy_size='large',
                phi_hidden_sizes='100x50',
                reg_scale=.0,
                lr_phi=0.0005,
                phi_obj='MinVar'):
     
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = epochs
        self.phi_epochs = phi_epochs
        self.lr = None # lr for policy neural network
        self.lr_phi = None # lr for phi function neural network
        self.lr_multiplier = 1.0  # dynamically adjust policy's lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy_size=policy_size
        self.phi_obj=phi_obj

        # create Phi networks
        self.reg_scale = reg_scale
        phi_hidden_sizes = [int(x) for x in phi_hidden_sizes.split("x")]
        self.phi = ContinousMLPPhiFunction(obs_dim, act_dim, 
                    hidden_sizes=phi_hidden_sizes, regular_scale=reg_scale)
        
        self.lr_phi = lr_phi

        self._build_graph()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
  
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            
            # Save only policy parameters
            policy_vars = tf.get_collection(\
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='policy_nn')
            
            var_dict = {}
            for var in policy_vars:
                logger.log(var.name)
                var_dict[var.name]=  var
            
            self._init_session()
            self.saver = tf.train.Saver(var_dict)

    
    def load_model(self, log_dir='log_dir/'):
        saver = tf.train.import_meta_graph(
                os.path.join(log_dir, 'policy_models/', 
                'policy.ckpt.meta'))
        
        saver.restore(self.sess, 
            tf.train.latest_checkpoint(
            os.path.join(log_dir, 'policy_models/')))

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        self.c_ph = tf.placeholder(tf.float32, (), 'c_ph')

        self.lr_phi_ph = tf.placeholder(tf.float32, (), 'eta_phi')

        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')



    def _policy_nn(self):
        
        with tf.variable_scope("policy_nn"):
            # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
            if self.policy_size == 'small':
                logger.log("using small structure")
                hid1_size = self.obs_dim # * 10
                hid3_size = self.act_dim # * 10
                hid2_size = int(np.sqrt(hid1_size * hid3_size))
                logvar_speed = (10 * hid3_size) // 48 
            elif self.policy_size == 'large':
                logger.log('Using large structure ')
                hid1_size = self.obs_dim  * 10
                hid3_size = self.act_dim  * 10
                hid2_size = int(np.sqrt(hid1_size * hid3_size))
                logvar_speed = (hid3_size) // 48 
            else:
                raise NotImplementedError
            
            self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
            
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph,
                        hid1_size, tf.tanh,
                        kernel_initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            
            out = tf.layers.dense(out, 
                        hid2_size, tf.tanh,
                        kernel_initializer= \
                        tf.random_normal_initializer( \
                        stddev=np.sqrt(1 / hid1_size)),
                        name="h2")
            
            out = tf.layers.dense(out, 
                        hid3_size, tf.tanh,
                        kernel_initializer= \
                        tf.random_normal_initializer( \
                        stddev=np.sqrt(1 / hid2_size)), 
                        name="h3")
            
            self.means = tf.layers.dense(out, self.act_dim,
                        kernel_initializer= \
                        tf.random_normal_initializer( \
                        stddev=np.sqrt(1 / hid3_size)),
                        name="means")

            logvar_speed = (10 * hid3_size) // 48

            log_vars = tf.get_variable('logvars', 
                            (logvar_speed, self.act_dim), 
                            tf.float32,
                            tf.constant_initializer(0.0))
            
            self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0

            self.policy_nn_vars = tf.get_collection(\
                        tf.GraphKeys.TRAINABLE_VARIABLES, 
                        scope='policy_nn')

            logger.log('Policy Params -- h1: {}, h2: {},\
                h3: {}, lr: {:.3g}, logvar_speed: {}'
                  .format(hid1_size, hid2_size, 
                  hid3_size, self.lr, logvar_speed))


    def _logprob(self):
        
        """ 
            Calculate log probabilities
            of a batch of observations & actions
        """
        
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(
                    tf.square(self.act_ph - self.means) /
                    tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(
                    tf.square(self.act_ph - self.old_means_ph) /
                    tf.exp(self.old_log_vars_ph), axis=1)
        
        self.logp_old = logp_old

    def _kl_entropy(self):  
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - \
                        log_det_cov_old + tr_old_new + \
                        tf.reduce_sum(tf.square(self.means - \
                        self.old_means_ph) / \
                        tf.exp(self.log_vars), \
                        axis=1) - self.act_dim)

        self.entropy = 0.5 * (self.act_dim * \
                    (np.log(2 * np.pi) + 1) + \
                    tf.reduce_sum(self.log_vars))


    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))


    def _loss_train_op(self):
   

        # get Phi function and its derivatives 
        phi_value, phi_act_g = self.phi(self.obs_ph, self.act_ph, reuse=False)
        self.phi_value = phi_value
        self.phi_act_g = phi_act_g
        self.phi_nn_vars = self.phi.phi_vars

        ll_mean_g = 1/tf.exp(self.log_vars) * (self.act_ph - self.means)
        ll_log_vars_g = -1/2 * ( 1/tf.exp(self.log_vars) \
                    - 1/tf.exp(self.log_vars) * \
                    (self.act_ph - self.means) * \
                    (self.act_ph - self.means) * \
                    1 / tf.exp(self.log_vars))

        self.phi_value.set_shape((None,))

        log_vars_inner = tf.expand_dims(tf.exp(self.logp - self.logp_old), 1) \
                        * (ll_log_vars_g * tf.expand_dims(self.advantages_ph 
                        - self.c_ph * self.phi_value, 1) \
                        + 1/2 * self.c_ph * ll_mean_g * self.phi_act_g )
            
        means_inner = tf.expand_dims(tf.exp(self.logp - self.logp_old), 1) \
                        * (ll_mean_g * tf.expand_dims(self.advantages_ph - 
                        self.c_ph * self.phi_value, 1) \
                        + self.c_ph * self.phi_act_g)
        
        loss1_log_vars = - tf.reduce_mean(
                        tf.stop_gradient(log_vars_inner) * \
                        tf.exp(self.log_vars)) 
        
        loss1_mean = -tf.reduce_mean(
                        tf.stop_gradient(means_inner) * \
                        self.means)
        
        loss1 = loss1_log_vars + loss1_mean
        
        loss2 = tf.reduce_mean(self.beta_ph * self.kl)
        
        loss3 = self.eta_ph * tf.square(\
                        tf.maximum(0.0, \
                        self.kl - 2.0 * self.kl_targ))

        self.loss = loss1 + loss2 + loss3
        
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss, 
                        var_list= self.policy_nn_vars)

        
        # phi loss train op
        if self.phi_obj == 'MinVar':
            means_mse = tf.reduce_sum(\
                    tf.reduce_mean( \
                    tf.square(means_inner - \
                    tf.reduce_mean(means_inner, \
                    axis=0)), axis = 0))
            
            logstd_vars_mse = tf.reduce_sum(\
                    tf.reduce_mean(\
                    tf.square(log_vars_inner - \
                    tf.reduce_mean(log_vars_inner,\
                    axis=0)), axis = 0))
                        
            gradient = tf.concat([means_inner, log_vars_inner], axis=1)

            est_A = tf.gather(gradient, tf.range(0, tf.shape(gradient)[0] //2))

            est_B = tf.gather(gradient, 
                    tf.range(tf.shape(gradient)[0] //2, 
                    tf.shape(gradient)[0]))
            
            # calculate loss
            est_var = tf.reduce_sum(\
                    tf.square(tf.reduce_mean(\
                    est_A, axis=0) - \
                    tf.reduce_mean(est_B, axis=0)))
        
        
        if self.reg_scale > 0.:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(
                        self.phi.kernel_regularizer, reg_variables)
           
            for var in reg_variables:
                logger.log("regularized, ", var.name, var.shape)
        else:
            reg_term = 0.

        if self.phi_obj == 'FitQ':
            self.phi_loss = tf.reduce_mean(\
                    tf.square(self.advantages_ph - \
                    self.phi_value), axis=0) + reg_term
            
            logger.log('phi_with FitQ as objective function')

        elif self.phi_obj == 'MinVar':
            
            self.phi_loss = means_mse + logstd_vars_mse + reg_term
            logger.log('phi with MinVar as objecive function')
        
        else:
            raise NotImplementedError
        
       
        phi_optimizer = tf.train.AdamOptimizer(self.lr_phi_ph)      
        self.phi_train_op = phi_optimizer.minimize(self.phi_loss, var_list=self.phi_nn_vars)

        self.means_inner = means_inner
        self.log_vars_inner = log_vars_inner

    def get_batch_gradient(self, observes, actions, advantages, c):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier,
                     self.lr_phi_ph: self.lr_phi,
                     self.c_ph:c}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        
        means_gradient, vars_gradient, phi_loss = self.sess.run(
                        [self.means_inner, 
                        self.log_vars_inner, self.phi_loss],
                        feed_dict=feed_dict)
        
        return {"mu_grad":means_gradient,
                'sigma_grad':vars_gradient, 
                'phi_loss':phi_loss}


    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, load_policy,  
            observes, actions, 
            advantages, use_lr_adjust, 
            ada_kl_penalty, c=1):
 
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier,
                     self.lr_phi_ph: self.lr_phi,
                     self.c_ph:c}

        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0

        # mini batch training
        self.sess.run(self.phi_train_op, feed_dict)
        
        if load_policy == 'save':
        
            for e in range(self.epochs):
                self.sess.run(self.train_op, feed_dict)
                loss, kl, entropy = self.sess.run([self.loss, 
                        self.kl, self.entropy], feed_dict)
                if kl > self.kl_targ * 4:  
                    break
          
            if (ada_kl_penalty):
                if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
                    self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
                    if (use_lr_adjust):
                        if self.beta > 30 and self.lr_multiplier > 0.1:
                            self.lr_multiplier /= 1.5
                elif kl < self.kl_targ / 2:
                    self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
                    if (use_lr_adjust):
                        if self.beta < (1 / 30) and self.lr_multiplier < 10:
                          self.lr_multiplier *= 1.5

            logger.record_dicts({
                'PolicyLoss': loss,
                'PolicyEntropy': entropy,
                'KL': kl,
                'Beta': self.beta,
                '_lr_multiplier': self.lr_multiplier})
        

    def save_policy(self, model_dir="models/policy_models"):
        self.saver.save(self.sess, 
                os.path.join(model_dir, 
                "policy.ckpt"))
