# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

################### introducing new op for mod
def np_mod(x,y):
    return (x % y).astype(np.float32)

def modgrad(op, grad):
    x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1] # the second argument
    return grad * 1, grad * tf.negative(tf.floordiv(x, y)) #the propagated gradient with respect to the first and second argument respectively

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

from tensorflow.python.framework import ops
def tf_mod(x,y, name=None):

    with ops.op_scope([x,y], name, "mod") as name:
        z = py_func(np_mod,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=modgrad)  # <-- here's the call to the gradient
        return z[0]

import Model_Factory.model_base as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

@tf.RegisterGradient("mod")
def _sub_grad(unused_op, grad):
  return grad, tf.negative(grad)

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)

    ############# CONV0_TWIN 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOutRes, prevExpandDim = model_base.conv_fire_parallel_module('conv0', images, kwargs.get('imageDepthChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  wd, **kwargs)
    # calc batch norm CONV1_TWIN
    if kwargs.get('batchNorm'):
        fireOutRes = model_base.batch_norm('batch_norm', fireOutRes, dtype)
    ############# CONV1_TWIN 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv1', fireOutRes, kwargs.get('imageDepthChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  wd, **kwargs)
    # calc batch norm CONV1_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV2_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv2', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  wd, **kwargs)
    # calc batch norm CONV2_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### RESIDUAL
    fireOut = tf.sub(fireOutRes, fireOut)
    ###### Pooling1 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool1')
    ############# CONV3_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv3', pool, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  wd, **kwargs)
    # calc batch norm CONV3_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV4_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv4', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  wd, **kwargs)
   # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### Pooling2 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    ############# CONV5 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv5', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[4]},
                                                         wd, **kwargs)
    # calc batch norm CONV5
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV6 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv6', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[5]},
                                                         wd, **kwargs)
    # calc batch norm CONV6
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### Pooling2 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV7 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv7', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[6]},
                                                         wd, **kwargs)
    # calc batch norm CONV7
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV8 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv8', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[7]},
                                                         wd, **kwargs)
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### DROPOUT after CONV8
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        #fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    ###### Prepare for fully connected layers
    # Reshape firout - flatten
    prevExpandDim = (kwargs.get('imageDepthRows')//(2*2*2))*(kwargs.get('imageDepthCols')//(2*2*2))*prevExpandDim
    fireOutFlat = tf.reshape(fireOut, [batchSize, -1])

    ############# FC1 layer with 1024 outputs
    fireOut, prevExpandDim = model_base.fc_fire_module('fc1', fireOutFlat, prevExpandDim,
                                                       {'fc': modelShape[8]},
                                                       wd, **kwargs)
    # calc batch norm FC1
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# FC2 layer with 8 outputs
    fireOut, prevExpandDim = model_base.fc_regression_module('fc2', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('outputSize')},
                                                             wd, **kwargs)
    return fireOut

def loss(pred, target, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pred, target, **kwargs)

def weighted_loss(tMatP, tMatT, **kwargs):
    mask = np.array([[100, 100, 100, 1, 100, 100, 100, 1, 100, 100, 100, 1]], dtype=np.float32)
    mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    tMatP = tf.multiply(mask, tMatP)
    tMatT = tf.multiply(mask, tMatT)
    return model_base.loss(tMatP, tMatT, **kwargs) 

def weighted_params_loss(targetP, targetT, **kwargs):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    targetP = tf_mod(targetP, mask)
    targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[1000, 1000, 1000, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return model_base.loss(targetP, targetT, **kwargs) 

def pcl_loss(pclA, tMatP, tMatT, **kwargs): # batchSize=Sne
    """
    Generate a ground truth point cloud using ground truth transformation
    Generate a prediction point cloud using predicted transformation
    L2 difference between ground truth and predicted point cloud is the loss value
    """
    # pclA, tMatP, tMatT are in batches
    # tMatP, tMatT should get a 0,0,0,1 row and be reshaped to 4x4
    tMatP = tf.concat([tMatP, tf.constant(np.repeat(np.array([[0, 0, 0, 1]],
                                                             dtype=np.float32),
                                                    kwargs.get('activeBatchSize'),
                                                    axis=0))],
                      1)
    tMatT = tf.concat([tMatT, tf.constant(np.repeat(np.array([[0, 0, 0, 1]],
                                                             dtype=np.float32),
                                                    kwargs.get('activeBatchSize'),
                                                    axis=0))],
                      1)
    tMatP = tf.reshape(tMatP, [kwargs.get('activeBatchSize'), 4, 4])
    tMatT = tf.reshape(tMatT, [kwargs.get('activeBatchSize'), 4, 4])
    # pclA should get a row of ones
    pclA = tf.concat([pclA, tf.constant(np.ones([kwargs.get('activeBatchSize'), 1, kwargs.get('pclCols')],
                                                dtype=np.float32))],
                     1)
    pclP = tf.matmul(tMatP, pclA)
    pclT = tf.matmul(tMatT, pclA)
    return model_base.loss(pclP, pclT, **kwargs)

def pcl_params_loss(pclA, pred, target, **kwargs): # batchSize=Sne
    """
    Generate transformation matrix using parameters for both prediction and ground truth
    Generate a ground truth point cloud using ground truth transformation
    Generate a prediction point cloud using predicted transformation
    L2 difference between ground truth and predicted point cloud is the loss value
    """
    print('paramsLoss')
    # pclA, tMatP, tMatT are in batches
    # tMatP, tMatT should get a 0,0,0,1 row and be reshaped to 4x4
    # transpose to easily extract columns: batchSize x 6 -> 6 x batchSize
    pred = tf.transpose(pred)
    target = tf.transpose(target)
    # generate tMatP and tMatT: 12 x batchSize
    a = pred[0]
    b = pred[1]
    g = pred[2]
    dx = pred[3]
    dy = pred[4]
    dz = pred[5]
    tMatP = tf.get_variable('tMatP', 12, initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
    tMatP = tf.Variable([
              tf.cos(a)*tf.cos(b), (tf.cos(a)*tf.sin(b)*tf.sin(g))-(tf.sin(a)*tf.cos(g)), (tf.cos(a)*tf.sin(b)*tf.cos(g))+(tf.sin(a)*tf.sin(g)), dx,
              tf.sin(a)*tf.cos(b), (tf.sin(a)*tf.sin(b)*tf.sin(g))+(tf.cos(a)*tf.cos(g)), (tf.sin(a)*tf.sin(b)*tf.cos(g))-(tf.cos(a)*tf.sin(g)), dy,
              -tf.sin(b),          tf.cos(b)*tf.sin(g),                                   tf.cos(b)*tf.cos(g),                                   dz
            ], dtype=tf.float32, name='tMatP', trainable=False)
    a = target[0]
    b = target[1]
    g = target[2]
    dx = target[3]
    dy = target[4]
    dz = target[5]
    tMatT = tf.get_variable('tMatT', 12, initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
    tMatT = tf.Variable([
              tf.cos(a)*tf.cos(b), (tf.cos(a)*tf.sin(b)*tf.sin(g))-(tf.sin(a)*tf.cos(g)), (tf.cos(a)*tf.sin(b)*tf.cos(g))+(tf.sin(a)*tf.sin(g)), dx,
              tf.sin(a)*tf.cos(b), (tf.sin(a)*tf.sin(b)*tf.sin(g))+(tf.cos(a)*tf.cos(g)), (tf.sin(a)*tf.sin(b)*tf.cos(g))-(tf.cos(a)*tf.sin(g)), dy,
              -tf.sin(b),          tf.cos(b)*tf.sin(g),                                   tf.cos(b)*tf.cos(g),                                   dz
           ], dtype=tf.float32, name='tMatT', trainable=False)
    # convert tMat's to correct form: 12 x batchSize -> batchSize x 12
    tMatP = tf.transpose(tMatP)
    tMatT = tf.transpose(tMatT)
    print('paramsLossDone')

    return pcl_loss(pclA, tMatP, tMatT, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)