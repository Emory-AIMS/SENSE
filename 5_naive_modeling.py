
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# In[ ]:


coeff_aggr = 1.0
coeff_fc = 10.0
coeff_hr = 10.0
coeff_sensor = 1.0
reg_scale = 1e-3
lr = 1e-8
epoch = 5000
batch_size = 2000
only_test = False
model_path = "./naive_model"


# # Training

# In[2]:


path = "./processed/train/"
label = np.load(path+"label_pca.npy")
static = np.load(path+"static_pca.npy")
hr = np.load(path+"hr_pca.npy")
sensor = np.load(path+"sensor_pca.npy")
label_ = 1 - label
label = np.concatenate([label_, label], axis=1)

print(label.shape)
print(static.shape)
print(hr.shape)
print(sensor.shape)


# In[3]:


neg_label = label[label[:, 0]==1, :]
neg_static = static[label[:, 0]==1, :]
neg_hr = hr[label[:, 0]==1, :, :]
neg_sensor = sensor[label[:, 0]==1, :, :]
print(neg_label.shape)
print(neg_static.shape)
print(neg_hr.shape)
print(neg_sensor.shape)

pos_weight = 1
pos_label = label[label[:, 1]==1, :] * pos_weight
pos_static = static[label[:, 1]==1, :]
pos_hr = hr[label[:, 1]==1, :, :]
pos_sensor = sensor[label[:, 1]==1, :, :]


print(pos_label.shape)
print(pos_static.shape)
print(pos_hr.shape)
print(pos_sensor.shape)

pos_loss_weight = neg_label.shape[0]*1.0/pos_label.shape[0]
# # over sampling

# In[ ]:


'''
sampling_rate = [6] * pos_label.shape[0]
pos_label = np.repeat(pos_label, sampling_rate, axis=0)
pos_static = np.repeat(pos_static, sampling_rate, axis=0)
pos_hr = np.repeat(pos_hr, sampling_rate, axis=0)
pos_sensor = np.repeat(pos_sensor, sampling_rate, axis=0)

print(pos_label.shape)
print(pos_static.shape)
print(pos_hr.shape)
print(pos_sensor.shape)


sampling_rate = [1] * neg_label.shape[0]
neg_label = np.repeat(neg_label, sampling_rate, axis=0)
neg_static = np.repeat(neg_static, sampling_rate, axis=0)
neg_hr = np.repeat(neg_hr, sampling_rate, axis=0)
neg_sensor = np.repeat(neg_sensor, sampling_rate, axis=0)

print(neg_label.shape)
print(neg_static.shape)
print(neg_hr.shape)
print(neg_sensor.shape)
'''


# In[ ]:


label = np.concatenate([pos_label, neg_label], axis=0)
static = np.concatenate([pos_static, neg_static], axis=0)
hr = np.concatenate([pos_hr, neg_hr], axis=0)
sensor = np.concatenate([pos_sensor, neg_sensor], axis=0)

print(label.shape)
print(static.shape)
print(hr.shape)
print(sensor.shape)


# In[ ]:


def weight_variable(shape, name, init_type="XV_1"):
    """
    """
    if init_type == "HE":
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False) # He
    elif init_type == "XV_1":
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True) # Xaiver 1
    elif init_type == "XV_2":
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False) # Xaiver 2
    elif init_type == "CONV":
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True) # Convolutional
    elif init_type == "VS":
        initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
    elif init_type == "TN":
        initializer = tf.truncated_normal_initializer(stddev=1.0)
    return tf.get_variable(initializer=initializer, shape=shape, name=name)

def bias_variable(shape, name):
    """
    """
    #initializer = tf.constant(0.1, shape=shape)
    initializer = tf.zeros_initializer()
    return tf.get_variable(initializer=initializer, shape=shape, name=name)


def fully_conn(x, weights, biases):
    return tf.add(tf.matmul(x, weights), biases)

def weighted_softmax_cross_entropy_with_logits(labels, logits, pos_weights):
	pos_weights = tf.ones([tf.shape(labels)[0], 1]) * pos_weights
	neg_weights = tf.ones([tf.shape(labels)[0], 1])
	weights = tf.concat([neg_weights, pos_weights], 1)
	return tf.nn.softmax_cross_entropy_with_logits(labels=labels*weights, logits=logits)

# # Model graph

# In[ ]:


tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # placeholder
    x_static = tf.placeholder(tf.float32, [None, 26])
    #x_hr = tf.placeholder(tf.float32, [None, 12, 40])
    x_hr = tf.placeholder(tf.float32, [None, 12, 10])
    #x_sensor = tf.placeholder(tf.float32, [None, 12, 270])
    x_sensor = tf.placeholder(tf.float32, [None, 12, 10])
    y = tf.placeholder(tf.float32, [None, 2])
    is_training = tf.placeholder(tf.bool, ())
    # FC model
    with tf.variable_scope("STATIC_FC"):
        FC_weight = tf.get_variable(initializer=tf.ones_initializer(), shape=(), name="FC_Weight")
        
        w_static_fc_0 = weight_variable([26, 64], "w_static_fc_0")
        b_static_fc_0 = bias_variable([64], "b_static_fc_0")
        w_static_fc_1 = weight_variable([64, 256], "w_static_fc_1")
        b_static_fc_1 = bias_variable([256], "b_static_fc_1")
        w_static_fc_2 = weight_variable([256, 1024], "w_static_fc_2")
        b_static_fc_2 = bias_variable([1024], "b_static_fc_2")
        w_static_fc_3 = weight_variable([1024, 256], "w_static_fc_3")
        b_static_fc_3 = bias_variable([256], "b_static_fc_3")
        w_static_fc_4 = weight_variable([256, 64], "w_static_fc_4")
        b_static_fc_4 = bias_variable([64], "b_static_fc_4")
        #
        w_static_fc_5 = weight_variable([64, 2], "w_static_fc_5")
        b_static_fc_5 = bias_variable([2], "b_static_fc_5")
        
        net = tf.nn.leaky_relu(fully_conn(x_static, w_static_fc_0, b_static_fc_0))
        net = tf.nn.leaky_relu(fully_conn(net, w_static_fc_1, b_static_fc_1))
        net = tf.nn.leaky_relu(fully_conn(net, w_static_fc_2, b_static_fc_2))
        net = tf.nn.leaky_relu(fully_conn(net, w_static_fc_3, b_static_fc_3)) # (?, 512)
        net_fc = tf.nn.leaky_relu(fully_conn(net, w_static_fc_4, b_static_fc_4)) # (?, 64)
        #
        logits_fc = fully_conn(net_fc, w_static_fc_5, b_static_fc_5)
        pred_fc = tf.nn.softmax(logits_fc)
        
        
        
    # LSTM
    with tf.variable_scope("HR_LSTM"):
        HR_weight = tf.get_variable(initializer=tf.ones_initializer(), shape=(), name="HR_Weight")
        #
        w_hr_0 = weight_variable([64, 2], "w_hr_0")
        b_hr_0 = bias_variable([2], "b_hr_0")
        
        # reshape
        x_hr_seq = tf.unstack(x_hr, axis=1) # shape: [batch_size, i, n_inputs], total num of i = n_steps
        hr_lstm_cell_0 = tf.nn.rnn_cell.LSTMCell(64)
        hr_lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(64)
        hr_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([hr_lstm_cell_0, hr_lstm_cell_1])
        # generate prediction
        #import pdb; pdb.set_trace()
        hr_outputs_seq, hr_states = tf.nn.static_rnn(hr_lstm_cell, x_hr_seq, dtype=tf.float32)
        hr_outputs = tf.stack(hr_outputs_seq, axis=1)
        hr_last_output = hr_outputs_seq[-1] # (?, 64)
        #
        #w_hr_lstm_0 = weight_variable([512, 64], "w_hr_lstm_0"); b_hr_lstm_0 = bias_variable([64], "b_hr_lstm_0")
        #net_hr = tf.nn.leaky_relu(fully_conn(hr_last_output, w_hr_lstm_0, b_hr_lstm_0)) # (?, 64)
        net_hr = hr_last_output
        
        #
        logits_hr = fully_conn(net_hr, w_hr_0, b_hr_0)
        pred_hr = tf.nn.softmax(logits_hr)
    # LSTM
    with tf.variable_scope("SENSOR_LSTM"):
        SEN_weight = tf.get_variable(initializer=tf.ones_initializer(), shape=(), name="SEN_Weight")
        #
        w_sensor_0 = weight_variable([64, 2], "w_sensor_0")
        b_sensor_0 = bias_variable([2], "b_sensor_0")
        
        # reshape
        x_sensor_seq = tf.unstack(x_sensor, axis=1) # shape: [batch_size, i, n_inputs], total num of i = n_steps
        sensor_lstm_cell_0 = tf.nn.rnn_cell.LSTMCell(128)
        sensor_lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(64)
        sensor_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([sensor_lstm_cell_0, sensor_lstm_cell_1])
        # generate prediction
        sensor_outputs_seq, sensor_states = tf.nn.static_rnn(sensor_lstm_cell, x_sensor_seq, dtype=tf.float32)
        sensor_outputs = tf.stack(sensor_outputs_seq, axis=1)
        sensor_last_output = sensor_outputs_seq[-1] # (?, 64)
        #
        w_sensor_lstm_0 = weight_variable([64, 64], "w_sensor_lstm_0")
        b_sensor_lstm_0 = bias_variable([64], "b_sensor_lstm_0")
        net_sensor = tf.nn.leaky_relu(fully_conn(sensor_last_output, w_sensor_lstm_0, b_sensor_lstm_0)) # (?, 64)
        
        #
        logits_sensor = fully_conn(net_sensor, w_sensor_0, b_sensor_0)
        pred_sensor = tf.nn.softmax(logits_sensor)
    # FC
    with tf.variable_scope("FC"):
        w_fc_0 = weight_variable([64*3, 64*3], "w_fc_0")
        b_fc_0 = bias_variable([64*3], "b_fc_0")
        w_fc_1 = weight_variable([64*3, 64*3], "w_fc_1")
        b_fc_1 = bias_variable([64*3], "b_fc_1")
        w_fc_2 = weight_variable([64*3, 64*3], "w_fc_2")
        b_fc_2 = bias_variable([64*3], "b_fc_2")
        w_fc_3 = weight_variable([64*3, 64], "w_fc_3")
        b_fc_3 = bias_variable([64], "b_fc_3")
        w_fc_4 = weight_variable([64, 2], "w_fc_4")
        b_fc_4 = bias_variable([2], "b_fc_4")
        #
        fc_in = tf.concat([FC_weight*net_fc, HR_weight*net_hr, SEN_weight*net_sensor], axis=1)
        net = tf.nn.leaky_relu(fully_conn(fc_in, w_fc_0, b_fc_0))
        net = tf.nn.leaky_relu(fully_conn(net, w_fc_1, b_fc_1))
        net = tf.nn.leaky_relu(fully_conn(net, w_fc_2, b_fc_2))
        net = tf.nn.leaky_relu(fully_conn(net, w_fc_3, b_fc_3))
        logits = fully_conn(net, w_fc_4, b_fc_4)
        pred = tf.nn.softmax(logits)
        
    with tf.variable_scope("LOSS"):
        loss = 0.0
        aggr_loss = tf.reduce_mean(weighted_softmax_cross_entropy_with_logits(labels=y, logits=logits, pos_weights=pos_loss_weight))
        loss += coeff_aggr * aggr_loss
        #
        loss_fc = tf.reduce_mean(weighted_softmax_cross_entropy_with_logits(labels=y, logits=logits_fc, pos_weights=pos_loss_weight))
        loss += coeff_fc * loss_fc
        loss_hr = tf.reduce_mean(weighted_softmax_cross_entropy_with_logits(labels=y, logits=logits_hr, pos_weights=pos_loss_weight))
        loss += coeff_hr * loss_hr
        loss_sensor = tf.reduce_mean(weighted_softmax_cross_entropy_with_logits(labels=y, logits=logits_sensor, pos_weights=pos_loss_weight))
        loss += coeff_sensor * loss_sensor
        
        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularize = tf.contrib.layers.l1_l2_regularizer(reg_scale, reg_scale)
        #print tf.GraphKeys.TRAINABLE_VARIABLES
        reg_term = sum([regularize(param) for param in opt_vars])
        loss += reg_term
        
    with tf.variable_scope("ACC"):
        correct_prediction = tf.equal(
            tf.argmax(y, 1),
            tf.argmax(pred, 1)
        )
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    with tf.variable_scope("OPT"):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)

        
        
def tf_load(sess, path, name='model.ckpt'):
    #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    if not os.path.exists(path):
        print("Wrong path: {}".format(path))
    saver.restore(sess, path +'/'+name)
    print("Restore model from {}".format(path +'/'+name))

def tf_save(sess, path, name='model.ckpt'):
    #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    if not os.path.exists(path):
        os.mkdir(path)
    saver.save(sess, path +'/'+name)
    print("Save model to {}".format(path +'/'+name))



if not only_test:
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config, graph=g) as sess:
	    sess.run(tf.global_variables_initializer())
	    for ep in range(epoch):
	        idx_list = list(range(label.shape[0]))
	        random.shuffle(idx_list)
	        n_batch = len(idx_list) // batch_size
	        for b_idx in range(n_batch):
	            batch_label = label[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :]
	            batch_static = static[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :]
	            #batch_hr = np.reshape(hr[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :], [-1, 12, 40])
	            batch_hr = hr[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :]
	            #batch_sensor = np.reshape(sensor[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :], [-1, 12, 270])
	            batch_sensor = sensor[idx_list[b_idx*batch_size:(b_idx+1)*batch_size], :]
	            
	            feed_dict = {
	                x_static: batch_static,
	                x_hr: batch_hr,
	                x_sensor: batch_sensor,
	                y: batch_label,
	                is_training: True
	            }
	            fetches = [train_op, loss, aggr_loss, loss_fc, loss_hr, loss_sensor, reg_term, acc, FC_weight, HR_weight, SEN_weight]
	            
	            _, model_loss, model_aggr_loss, model_fc_loss, model_hr_loss, model_sen_loss, model_reg, model_acc, fc_weight, hr_weight, sen_weight = sess.run(fetches=fetches, feed_dict=feed_dict)
	        print("epoch: ", ep)
	        print("fc weight: {:.4f}, hr weight: {:.4f}, sen weight: {:.4f}".format(fc_weight, hr_weight, sen_weight))
	        print("model loss: {:.4f}, model accuracy: {:.4f}".format(model_loss, model_acc))
	        print("model aggr loss: {:.4f}, model fc loss: {:.4f}, model hr loss: {:.4f}, model sen loss: {:.4f}, model reg loss: {:.4f}".format(
	        	model_aggr_loss, model_fc_loss, model_hr_loss, model_sen_loss, model_reg))
	        print()
	        
	    tf_save(sess, model_path)
        


# # Test

# In[ ]:


path = "./processed/test/"
label = np.load(path+"label_pca.npy")
static = np.load(path+"static_pca.npy")
hr = np.load(path+"hr_pca.npy")
sensor = np.load(path+"sensor_pca.npy")

label_ = 1 - label
label = np.concatenate([label_, label], axis=1)
print(label.shape)
print(static.shape)
print(hr.shape)
print(sensor.shape)


# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    tf_load(sess, model_path)
    
    batch_label = label
    batch_static = static
    #batch_hr = np.reshape(hr, [-1, 12, 40])
    batch_hr = hr
    #batch_sensor = np.reshape(sensor, [-1, 12, 270])
    batch_sensor = sensor

    feed_dict = {
        x_static: batch_static,
        x_hr: batch_hr,
        x_sensor: batch_sensor,
        y: batch_label,
        is_training: False
    }
    fetches = [loss, acc, pred]

    model_loss, model_acc, model_pred_probs = sess.run(fetches=fetches, feed_dict=feed_dict)
    auc = roc_auc_score(batch_label, model_pred_probs)
    print("model loss: {}, model accuracy: {}, model AUC: {}".format(model_loss, model_acc, auc))
    print()
    # neg
    batch_label = label[label[:, 0]==1, :]
    batch_static = static[label[:, 0]==1, :]
    #batch_hr = np.reshape(hr[label[:, 0]==1, :, :], [-1, 12, 40])
    batch_hr = hr[label[:, 0]==1, :, :]
    #batch_sensor = np.reshape(sensor[label[:, 0]==1, :, :], [-1, 12, 270])
    batch_sensor = sensor[label[:, 0]==1, :, :]

    feed_dict = {
        x_static: batch_static,
        x_hr: batch_hr,
        x_sensor: batch_sensor,
        y: batch_label,
        is_training: False
    }
    fetches = [loss, acc]

    model_loss, model_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
    print("neg loss: {}, neg accuracy: {}".format(model_loss, model_acc))
    print()
    # pos
    batch_label = label[label[:, 1]==1, :]
    batch_static = static[label[:, 1]==1, :]
    #batch_hr = np.reshape(hr[label[:, 1]==1, :, :], [-1, 12, 40])
    batch_hr = hr[label[:, 1]==1, :, :]
    #batch_sensor = np.reshape(sensor[label[:, 1]==1, :, :], [-1, 12, 270])
    batch_sensor = sensor[label[:, 1]==1, :, :]

    feed_dict = {
        x_static: batch_static,
        x_hr: batch_hr,
        x_sensor: batch_sensor,
        y: batch_label,
        is_training: False
    }
    fetches = [loss, acc]

    model_loss, model_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
    print("pos loss: {}, pos accuracy: {}".format(model_loss, model_acc))
    print()
    
        


# In[ ]:




