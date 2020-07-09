import tensorflow as tf


def cnn_3d(sequence, is_training):
    
    with tf.name_scope('block1'):
        x = tf.layers.conv3d(inputs=sequence, filters=32, kernel_size=[3, 5, 5],
                                 strides=[1, 2, 2], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=FLAGS.dropout_rate, training=True)
        
    with tf.name_scope('block2'):
        x = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=128, kernel_size=[3, 3, 3],
                                 strides=[1, 2, 2], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool3d(x, ksize=[1, 3, 1, 1, 1], strides=[1, 2, 1, 1, 1],
                             padding='SAME')
        x = tf.layers.dropout(x, rate=FLAGS.dropout_rate, training=True)
        
    with tf.name_scope('block3'):
        x = tf.layers.conv3d(inputs=x, filters=128, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=128, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=256, kernel_size=[3, 3, 3],
                                 strides=[1, 2, 2], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=FLAGS.dropout_rate, training=True)
        
    with tf.name_scope('block4'):
        x = tf.layers.conv3d(inputs=x, filters=256, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=256, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=512, kernel_size=[3, 3, 3],
                                 strides=[1, 2, 2], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=FLAGS.dropout_rate, training=True)
        
    with tf.name_scope('block5'):
        x = tf.layers.conv3d(inputs=x, filters=512, kernel_size=[3, 3, 3],
                                 strides=[1, 1, 1], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(inputs=x, filters=512, kernel_size=[3, 3, 3],
                                 strides=[1, 2, 2], dilation_rate=[1, 1, 1],
                                 padding='same', data_format='channels_last')
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.nn.relu(x)
        print(x)
    with tf.name_scope('global_avg_pooling'):
        x = tf.reduce_mean(x, axis=[4])
        print(x)
        
    with tf.name_scope('fully_con'):
        x = tf.layers.flatten(x)
        print(x)
        x = tf.layers.dense(x, units=NUM_CLASSES)
        print(x)
        
    return x

