from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib

def conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2d"):
  """
  Args:
    inputs: nhwc
    kernel_shape: [height, width]
    mask_type: None or 'A' or 'B' or 'C'
  Returns:
    outputs: nhwc
  """
  #scope is just a name used to share variables. Don't worry about it, it isn't being used in our code.
  with tf.variable_scope(scope) as scope:
    #get kernel height and width from kernel shape
    kernel_h, kernel_w = kernel_shape
    #get stride hight and width
    stride_h, stride_w = strides
    #get other hyperparams
    batch_size, height, width, in_channel = inputs.get_shape().as_list()


    center_h = kernel_h // 2
    center_w = kernel_w // 2
    #the dimentions of the convolutional kernel must be an odd number
    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width must be odd number"
    mask = np.zeros((kernel_h, kernel_w, in_channel, num_outputs), dtype=np.float32)
    if mask_type is not None:
      #C
      mask[:center_h, :, :, :] = 1
      if mask_type == 'A':
        mask[center_h, :center_w, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, 0:in_channel:3, 2:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 2:num_outputs:3] = 1
        """
      if mask_type == 'B':
        mask[center_h, :center_w+1, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #R Channel
        mask[center_h, center_w, 0:in_channel:3, 0:num_outputs:3] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, :, 2:num_outputs:3] = 1
        """
    else:
      mask[:, :, :, :] = 1

    weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    weights = weights * mask
    biases = tf.get_variable("biases", [num_outputs],
          tf.float32, tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding="SAME")
    outputs = tf.nn.bias_add(outputs, biases)

    return outputs

def gated_conv2d(inputs, state, kernel_shape, scope):
  """
  Args:
    inputs: nhwc
    state:  nhwc
    kernel_shape: [height, width]
  Returns:
    outputs: nhwc
    new_state: nhwc
  """
  with tf.variable_scope(scope) as scope:
    batch_size, height, width, in_channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape
    #state route
    left = conv2d(state, 2 * in_channel, kernel_shape, strides=[1, 1], mask_type='C', scope="conv_s1")
    left1 = left[:, :, :, 0:in_channel]
    left2 = left[:, :, :, in_channel:]
    left1 = tf.nn.tanh(left1)
    left2 = tf.nn.sigmoid(left2)
    new_state = left1 * left2
    left2right = conv2d(left, 2 * in_channel, [1, 1], strides=[1, 1], scope="conv_s2")
    #input route
    right = conv2d(inputs, 2 * in_channel, [1, kernel_w], strides=[1, 1], mask_type='B', scope="conv_r1")
    right = right + left2right
    right1 = right[:, :, :, 0:in_channel]
    right2 = right[:, :, :, in_channel:]
    right1 = tf.nn.tanh(right1)
    right2 = tf.nn.sigmoid(right2)
    up_right = right1 * right2
    up_right = conv2d(up_right, in_channel, [1, 1], strides=[1, 1], mask_type='B', scope="conv_r2")
    outputs = inputs + up_right

    return outputs, new_state

def batch_norm(x, train=True, scope=None):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

def resnet_block(lr_image_input, inputs, num_outputs, kernel_shape, strides=[1, 1], scope=None, train=True):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
  Returns:
    outputs: nhw(num_outputs)
  """
  with tf.variable_scope(scope) as scope:
    conv1 = conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv1")
    bn1 = batch_norm(conv1, train=train, scope='bn1')
    # sp1 = spade_resblock(lr_image_input, bn1, num_outputs, use_bias=True, sn=False, scope='spade_resblock')
    relu1 = tf.nn.relu(sp1)
    conv2 = conv2d(relu1, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2")
    bn2 = batch_norm(conv2, train=train, scope='bn2')
    sp2 = spade_resblock(lr_image_input, bn2, num_outputs, use_bias=True, sn=False, scope='spade_resblock')
    output = inputs + sp2

    return output

def deconv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], scope="deconv2d"):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
    strides: [stride_h, stride_w]
  Returns:
    outputs: nhwc
  """
  with tf.variable_scope(scope) as scope:
    return tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_shape, strides, \
          padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.1), \
          biases_initializer=tf.constant_initializer(0.0))

################################# SPADE resnet_block ##########################33
# tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = None

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    if uniform :
        factor = gain * gain
        mode = 'FAN_AVG'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_AVG'

    return factor, mode, uniform

factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
reuse = False

# def resnet_block(lr_image_input, inputs, num_outputs, kernel_shape, strides=[1, 1], scope=None, train=True):
def spade_resblock(segmap, x_init, channels, use_bias=True, sn=False, scope=None, train=True):
# def spade_resblock(segmap, x_init, channels, use_bias=True, sn=False, scope='spade_resblock'):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.variable_scope(scope) :
        x = spade(segmap, x_init, channel_in, use_bias=use_bias, sn=False, scope='spade_1')
        x = lrelu(x, 0.2)
        # x = conv2d(x, channel_middle, [3, 3], strides=[1, 1], mask_type=None, scope="conv_1")
        x = conv_spade(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1', reuse=None)

        x = spade(segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='spade_2')
        x = lrelu(x, 0.2)
        # x = conv2d(x, channel_middle, [3, 3], strides=[1, 1], mask_type=None, scope="conv_2")
        x = conv_spade(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2', reuse=None)

        if channel_in != channels :
            x_init = spade(segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='spade_shortcut')
            # x = conv2d(x, channels, [1, 1], strides=[1, 1], mask_type=None, scope="conv_shortcut")
            x_init = conv_spade(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut', reuse=None)

        return x + x_init
        # return x

def spade(segmap, x_init, channels, use_bias=True, sn=False, scope='spade') :
    with tf.variable_scope(scope) :
        x = param_free_norm(x_init)
        # print("x- shape", x.get_shape())
        # print("segmap- shape", segmap.get_shape())

        _, x_h, x_w, _ = x_init.get_shape().as_list()

        x_size=[x_h, x_w]
        segmap = tf.image.resize_nearest_neighbor(segmap, x_size)
        # print("segmap- shape", segmap.get_shape())

        _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()
        factor_h = segmap_h // x_h  # 256 // 4 = 64
        factor_w = segmap_w // x_w

        if(factor_h==0 or factor_w==0):
            segmap_down = segmap
        else:
            segmap_down = down_sample(segmap, factor_h, factor_w)
        # segmap_down = conv2d(segmap_down, 128, [5, 5], strides=[1, 1], mask_type=None, scope="conv_down")
        segmap_down = conv_spade(segmap_down, channels=128, kernel=5, stride=1, pad=2, pad_type='zero', use_bias=use_bias, sn=sn, scope=scope + 'conv_128', reuse=None)

        segmap_down_1 = tf.nn.relu(segmap_down)
        segmap_down_2 = tf.nn.relu(segmap_down)
        # segmap_gamma = conv2d(segmap_down, channels, [5, 5], strides=[1, 1], mask_type=None, scope="conv_gamma")
        # segmap_beta = conv2d(segmap_down, channels, [5, 5], strides=[1, 1], mask_type=None, scope="conv_beta")
        segmap_gamma = conv_spade(segmap_down_1, channels=channels, kernel=5, stride=1, pad=2, pad_type='zero', use_bias=use_bias, sn=sn, scope=scope+'conv_gamma', reuse=None)
        segmap_beta = conv_spade(segmap_down_2, channels=channels, kernel=5, stride=1, pad=2, pad_type='zero', use_bias=use_bias, sn=sn, scope=scope+'conv_beta', reuse=None)
        #
        # print("beta",segmap_beta.get_shape())
        # print("gamma", segmap_gamma.get_shape())
        # print("x", x.get_shape())
        x = x * (1 + segmap_gamma) + segmap_beta

        return x

def param_free_norm(x, epsilon=1e-5) :
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x_std = tf.sqrt(x_var + epsilon)

    return (x - x_mean) / x_std

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha*tf.nn.relu(-x)

def conv_spade(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)
        return x

########################################################## SPADE ################
