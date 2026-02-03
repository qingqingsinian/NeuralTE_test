"""

	Actor of Deep Deterministic policy gradient

"""

import tflearn
import tensorflow as tf


class ActorNetwork:
    def __init__(self, sess, dim_state, dim_action, bound_action, learning_rate, tau, num_path):
        self.__sess = sess
        self.__dim_s = dim_state
        self.__dim_a = dim_action
        self.__max_a = bound_action
        self.__learning_rate = learning_rate
        self.__num_path = num_path
        self.__tau = tau

        # performance network
        cur_para_num = len(tf.trainable_variables())
        self.__input, self.__out, self.__out_scaled = self.buildNetwork()
        self.__paras = tf.trainable_variables()[cur_para_num:]

        # Target network
        self.__target_input, self.__target_out, self.__target_out_scaled = self.buildNetwork()
        self.__target_paras = tf.trainable_variables()[(len(self.__paras) + cur_para_num):]

        # update parameters of target network
        self.__ops_update_target = []
        for i in range(len(self.__target_paras)):
            val = tf.add(tf.multiply(self.__paras[i], self.__tau), tf.multiply(self.__target_paras[i], 1. - self.__tau))
            op = self.__target_paras[i].assign(val)
            self.__ops_update_target.append(op)

        # provided by Critic
        self.__gradient_action = tf.placeholder(tf.float32, [None, self.__dim_a])

        # calculate gradients
        self.__actor_gradients = tf.gradients(self.__out_scaled, self.__paras, -self.__gradient_action)

        self.__optimize = tf.train.AdamOptimizer(self.__learning_rate) \
            .apply_gradients(zip(self.__actor_gradients, self.__paras))

        self.__num_trainable_vars = len(self.__paras) + len(self.__target_paras)

    @property
    def session(self):
        return self.__sess

    @property
    def num_trainable_vars(self):
        return self.__num_trainable_vars

    @property
    def dim_state(self):
        return self.__dim_s

    @property
    def dim_action(self):
        return self.__dim_a

#build ResNet-based(more complex) critic network
#
    def myConv1d_first(self, input_2d, my_filter_size, stride):
        #input_2d = tf.expand_dims(input_1d, 1)
        #print(input_2d.op.name, '', input_2d.get_shape().as_list())        
        input_3d = tf.expand_dims(input_2d, 3)
        print(input_3d.op.name, '', input_3d.get_shape().as_list())    
        
#        net = tf.layers.conv1d(input_2d, out_dim, my_filter_size, stride, padding='SAME', activation=tf.nn.relu)          
        net = tf.nn.conv2d(input_3d, my_filter_size, stride, padding='SAME', data_format="NHWC") 
        print(net.op.name, '', net.get_shape().as_list())        
        #net = tf.squeeze(net, [2])          
        #print(net.op.name, '', net.get_shape().as_list())        
        #print(net)                                         
        return net

    def myConv1d(self, input_3d, my_filter_size, stride):
        net = tf.nn.conv2d(input_3d, my_filter_size, stride, padding='SAME', data_format="NHWC")                    
        print(net.op.name, '', net.get_shape().as_list())                                                 
        return net

    def myConv1d_last(self, input_3d, my_filter_size, stride):
        #input_2d = tf.expand_dims(input_1d, 2)
        #print(input_2d.op.name, '', input_2d.get_shape().as_list())  
        net = tf.nn.conv2d(input_3d, my_filter_size, stride, padding='SAME', data_format="NHWC")          
        print(net.op.name, '', net.get_shape().as_list())        
        net = tf.squeeze(net, [1, 3])          
        print(net.op.name, '', net.get_shape().as_list())        
        #print(net)                                         
        return net

    def buildNetwork(self):
        inputs = tf.placeholder(tf.float32, [None, self.__dim_s])    
        _inputs = tf.reshape(inputs, [tf.shape(inputs)[0], 1, self.__dim_s])
        print(_inputs.op.name, '', inputs.get_shape().as_list()) 
                 
        net = _inputs
        # for more or less parameters
        #net = tf.contrib.layers.fully_connected(net, 128, activation_fn=tf.nn.leaky_relu)
        print("*******This is actor network building!*******")
      
        filter = tf.Variable(tf.random_normal([1, 4, 1, 4]))        
        net = self.myConv1d_first(net, filter, [1, 1, 1, 1])
        net = tf.nn.relu(net)                 
        #filter = tf.get_variable('weight-e', [1, 2, 4, 1], 
                 #initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        filter = tf.Variable(tf.random_normal([1, 4, 4, 2]))        
        net = self.myConv1d(net, filter, [1, 1, 1, 1])
        net = tf.nn.relu(net)        
        filter = tf.Variable(tf.random_normal([1, 2, 2, 1]))         
        net = self.myConv1d_last(net, filter, [1, 1, 1, 1])
        net = tf.nn.relu(net)
        
#add shotcut
        net = tf.add(net, inputs)
                
        # as original paper indicates
        w_init = tflearn.initializations.uniform(minval=-3e-3, maxval=3e-3)

        out_vec = []
        for i in self.__num_path:
            out = tf.contrib.layers.fully_connected(net, i, activation_fn=tf.nn.softmax, weights_initializer=w_init)
            out_vec.append(out)
        out = tf.concat([i for i in out_vec], axis=1)

        out_scaled = tf.multiply(out, self.__max_a)

        print("inputs's original demension is:")
        print(inputs.get_shape().as_list())         
        print("out's original demension is:")
        print(out.get_shape().as_list())          
        print("out's original demension is:")
        print(out_scaled.get_shape().as_list())  

        return inputs, out, out_scaled
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def train(self, inputs, gradient_action):
        self.__sess.run(self.__optimize, feed_dict={
            self.__input: inputs,
            self.__gradient_action: gradient_action
        })

    def predict(self, inputs):
        return self.__sess.run(self.__out_scaled, feed_dict={
            self.__input: inputs
        })

    def predict_target(self, inputs):
        return self.__sess.run(self.__target_out_scaled, feed_dict={
            self.__target_input: inputs
        })

    def update_target_paras(self):
        self.__sess.run(self.__ops_update_target)
