from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import re

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def clstm_block(input_tensor, nb_hidden, ks1, ks2, pooling,
                batch_normalization, return_sequences):
    """
    x: input tensor
    nb_hidden: int
    ks: int
    pooling: str 'max'|'avg'
    batch_normalization: bool
    return_sequences: bool
    """
    # Kernel regularizer
    reg = tf.keras.regularizers.l2(FLAGS.kernel_regularizer)
    # ConvLSTM2D layer
    clstm_output = tf.keras.layers.ConvLSTM2D(filters=nb_hidden,
                                   kernel_size=(ks1,ks2),
                                   padding=FLAGS.padding,
                                   strides=(FLAGS.strides,FLAGS.strides),
                                   kernel_regularizer=reg,
                                   dropout=FLAGS.dropout_rate,
                                   return_sequences=return_sequences)(input_tensor)
    if return_sequences == True:
        # Maxpooling layer per time slice
        if pooling == 'max':
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(inputs=clstm_output)
        if pooling == 'avg':
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D())(inputs=clstm_output)
    else:
        if pooling == 'max':
            x = tf.keras.layers.MaxPooling2D()(inputs=clstm_output)
        if pooling == 'avg':
            x = tf.keras.layers.AveragePooling2D()(inputs=clstm_output)
    print(x)
    # Normalize according to batch statistics
    if batch_normalization:
        x = tf.layers.batch_normalization(inputs=x)
        print(x)
    return x, clstm_output

    
def clstm_gap(sequence, bn):

    layers = ast.literal_eval(FLAGS.layers)
    rs = ast.literal_eval(FLAGS.return_sequences)
    nb_clstm_layers = len(layers)
    
    print(x)

    for l in range(nb_clstm_layers):
        name_scope = 'block' + str(l + 1)
        with tf.name_scope(name_scope):
            x = clstm_block(x, nb_hidden=layers[l], ks=FLAGS.kernel_size,
                            pooling=True, batch_normalization=bn,
                            return_sequences=rs[l])


    with tf.name_scope('gap'):
        x = tf.nn.avg_pool3d(x, ksize=[1,16,1,1,1],
                             strides=[1,1,1,1,1],
                             padding='VALID')
        print(x)
        x = tf.layers.conv3d(inputs=x, filters=NUM_CLASSES, kernel_size=[1, 1, 1],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='valid', data_format='channels_last')
        # x = tf.nn.relu(x)
        # x = tf.layers.dense(x, units=NUM_CLASSES)
        print(x)
        x = tf.reshape(x, [-1, NUM_CLASSES])
        print(x)

    return x


def clstm(x, bn, num_classes):
    """x: 5D tensor, sequence of images
       bn: bool, whether to batch normalize
       return: x, the transformed input sequence."""

    layers = ast.literal_eval(FLAGS.layers)
    rs = ast.literal_eval(FLAGS.return_sequences)
    nb_clstm_layers = len(layers)
    
    print(x)

    for l in range(nb_clstm_layers):
        name_scope = 'block' + str(l + 1)
        with tf.name_scope(name_scope):
            x, clstm_output = clstm_block(x, nb_hidden=layers[l],
                            ks1=FLAGS.kernel_size_1,
                            ks2=FLAGS.kernel_size_2,
                            pooling=FLAGS.pooling_method, batch_normalization=bn,
                            return_sequences=rs[l])
            print('x: ', x)
            print('clstm_output: ', clstm_output)


    with tf.name_scope('fully_con'):
        if FLAGS.only_last_element_for_fc == 'yes':
            # Only pass on the last element of the sequence to FC.
            # return_seq is True just to save it in the graph for gradcam.
            x = tf.layers.flatten(x[:,-1,:,:,:])
        else:
            x = tf.layers.flatten(x)
        print(x)
        x = tf.layers.dense(x, units=num_classes)
        print(x)

    return x, clstm_output

