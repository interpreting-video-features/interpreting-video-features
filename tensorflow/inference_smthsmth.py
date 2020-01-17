import tensorflow as tf
from sklearn.metrics import classification_report
from keras.utils import np_utils
from PIL import Image
import pandas as pd
import numpy as np
import ast
import cv2
import os

CLIP_LENGTH = 16
NUM_CLASSES = 174


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(tf.flags.FLAGS)

NUM_CLASSES = 174
lr = 0.001

tf.app.flags.DEFINE_string('f', '', 'kernel')

tf.app.flags.DEFINE_integer('nb_epochs',
    10,
    """Number of epochs to train.""")
tf.app.flags.DEFINE_integer('nb_train_samples',
    168913,
    """The number of samples in the training set.""")
tf.app.flags.DEFINE_integer('nb_val_samples',
    24777,
    """The number of samples in the validation set.""")
tf.app.flags.DEFINE_integer('nb_hidden',
    32,
    """Number of hidden units in a CLSTM-layer.""")
tf.app.flags.DEFINE_integer('nb_clstm_layers',
    3,
    """Number of CLSTM-layers.""")
tf.app.flags.DEFINE_integer('seq_length',
    16,
    """Number of frames per sequence.""")
tf.app.flags.DEFINE_integer('image_size',
    224,
    """Square image size.""")
tf.app.flags.DEFINE_integer('shuffle_buffer',
    169000,
    """Shuffle buffer.""")
tf.app.flags.DEFINE_integer('batch_size',
    1,
    """Number of sequences per batch.""")
tf.app.flags.DEFINE_integer('strides',
    2,
    """x sized strides for C-LSTM kernel (x,x).""")
tf.app.flags.DEFINE_string('padding',
    'valid',
    """Padding for the C-LSTM convolution. 'same' | 'valid' """)
tf.app.flags.DEFINE_float('dropout_rate',
    0.0,
    """The rate at which to perform dropout (how much to drop).""")
tf.app.flags.DEFINE_float('momentum',
    0.2,
    """Momentum for the gradient descent.""")
tf.app.flags.DEFINE_float('weight_decay',
    0.00001,
    """Weight decay (decoupled right now).""")
tf.app.flags.DEFINE_integer('nb_parallel_calls',
    16,
    """Number of parallel core calls when preparing a batch.""")
tf.app.flags.DEFINE_integer('lr_decay_patience',
    2,
    """Number of epochs without improvement to wait before decreased learning rate.""")
tf.app.flags.DEFINE_string('shuffle_data',
    'no',
    """Whether to shuffle the training data. See SHUFFLE_BUFFER param.""")
tf.app.flags.DEFINE_string('train_data',
    '/data/tfrecords/train.tfrecords',
    """Path to training data, in .tfrecords format.""")
tf.app.flags.DEFINE_string('val_data',
    '/data/tfrecords/validation.tfrecords',
    """Path to validation data, in .tfrecords format.""")
tf.app.flags.DEFINE_string('checkpoint_name',
    'model.ckpt',
    """To go in checkpoints/model.ckpt""")
tf.app.flags.DEFINE_string('output_folder',
    None,
    """Where to save predictions as .npy files""")
tf.app.flags.DEFINE_string('test_run',
    'no',
    """Whether to only do a test run with few steps. 'yes' if so.""")
tf.app.flags.DEFINE_string('model',
    'clstm',
    """Which model to run with. cnn_3d | clstm""")
tf.app.flags.DEFINE_string('optimizer',
    'momentum',
    """Which optimizer to run with. adadelta | momentum (with decoupled w d)""")
tf.app.flags.DEFINE_float('learning_rate_start',
    0.001,
    """Learning rate to start with.""")
tf.app.flags.DEFINE_float('learning_rate_end',
    0.001,
    """Minimum learning rate after decay.""")
tf.app.flags.DEFINE_float('kernel_regularizer',
    0.01,
    """Kernel regularizer for the ConvLSTM2D layer.""")
tf.app.flags.DEFINE_string('clstm_only_last_output',
    'no',
    """Whether to only use the last element of the sequence or\
       return the full sequence at the last layer of the C-LSTM.""")

print(tf.app.flags.FLAGS.flag_values_dict())

FLAGS = tf.app.flags.FLAGS


def clstm(sequence):
    with tf.name_scope('block1'):
        print(sequence)
        x = tf.keras.layers.ConvLSTM2D(filters=FLAGS.nb_hidden, kernel_size=(5,5),
                                       padding=FLAGS.padding,
                                       strides=(FLAGS.strides,FLAGS.strides),
                                       kernel_regularizer=tf.keras.regularizers.l2(FLAGS.kernel_regularizer),
                                       return_sequences=True)(sequence)
        
        x = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)(x)
        
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(inputs=x)
        
        x = tf.layers.batch_normalization(inputs=x)
        
    if FLAGS.nb_clstm_layers >= 2:
        with tf.name_scope('block2'):
            x = tf.keras.layers.ConvLSTM2D(filters=FLAGS.nb_hidden, kernel_size=(5,5),
                                           padding=FLAGS.padding,
                                           strides=(FLAGS.strides,FLAGS.strides),
                                           kernel_regularizer=tf.keras.regularizers.l2(FLAGS.kernel_regularizer),
                                           return_sequences=True)(x)
            
            x = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)(x)
            
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(inputs=x)
            
            x = tf.layers.batch_normalization(inputs=x)
            
    if FLAGS.nb_clstm_layers >= 3:
        with tf.name_scope('block3'):
            clstm_3 = tf.keras.layers.ConvLSTM2D(filters=FLAGS.nb_hidden, kernel_size=(5,5),
                                           padding=FLAGS.padding,
                                           strides=(FLAGS.strides,FLAGS.strides),
                                           kernel_regularizer=tf.keras.regularizers.l2(FLAGS.kernel_regularizer),
                                           return_sequences=True)(x)
            
            x = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)(clstm_3)
            
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(inputs=x)
            
            x = tf.layers.batch_normalization(inputs=x)
            
    if FLAGS.nb_clstm_layers >= 4:
        with tf.name_scope('block4'):
            if FLAGS.clstm_only_last_output == 'yes':
                x = tf.keras.layers.ConvLSTM2D(filters=FLAGS.nb_hidden, kernel_size=(5,5),
                                               padding=FLAGS.padding,
                                               kernel_regularizer=tf.keras.regularizers.l2(FLAGS.kernel_regularizer),
                                               strides=(FLAGS.strides,FLAGS.strides))(x)
                
                x = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)(x)
                
                x = tf.keras.layers.MaxPooling2D()(inputs=x)
                
                x = tf.layers.batch_normalization(inputs=x)
                
            else:
                
                x = tf.keras.layers.ConvLSTM2D(filters=FLAGS.nb_hidden, kernel_size=(5,5),
                                               padding=FLAGS.padding,
                                               strides=(FLAGS.strides,FLAGS.strides),
                                               kernel_regularizer=tf.keras.regularizers.l2(FLAGS.kernel_regularizer),
                                               return_sequences=True)(x)
                
                x = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)(x)
                
                x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(inputs=x)
                
                x = tf.layers.batch_normalization(inputs=x)
                

    with tf.name_scope('fully_con'):
        x = tf.layers.flatten(x)
        
        x = tf.layers.dense(x, units=NUM_CLASSES)
        

    return x, clstm_3


def collapse_batches_of_y(y):
    nb_batches = len(y)
    for b in range(nb_batches-1):
        if b == 0:
            collapsed_y = np.concatenate((y[b], y[b+1]), axis=0)
            continue
        collapsed_y = np.concatenate((collapsed_y, y[b+1]), axis=0)
    return collapsed_y


def parse_fn(proto):

    # Define the tfrecord again. The sequence was saved as a string.
    keys_to_features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'nb_frames': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'frames': tf.VarLenFeature(tf.string),
    }

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Indices across sequence
    frame_inds = tf.range(FLAGS.seq_length)  
    
    # Decode the image strings
    images = tf.map_fn(lambda i: \
                       tf.cast(tf.image.decode_jpeg(
                        parsed_features["frames"].values[i]),
                       dtype=tf.int32),
                       frame_inds,
                       dtype=tf.int32)

    images = tf.cast(images, tf.float32)
    # UNCOMMENT TO TEST REVERSE PERFORMANCE
    # images = tf.reverse(tensor=images, axis=[0])
    label = tf.one_hot(parsed_features['label'], NUM_CLASSES)
    label = tf.cast(label, tf.int32)

    return images, label

def create_dataset(filepath):
    import os.path
    dataset = tf.data.TFRecordDataset(filepath)
    print(filepath)
    print(os.path.isfile(filepath))
    if FLAGS.shuffle_data == 'yes':
        print('Shuffling the training data...')
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
          map_func=parse_fn,
          batch_size=FLAGS.batch_size,
          num_parallel_calls=FLAGS.nb_parallel_calls)) \
    .prefetch(FLAGS.batch_size)

    return dataset


def get_top_k(array, k):
    top_k = array.argsort()[-k:][::-1]
    return top_k

assert FLAGS.output_folder.endswith('/')

# First we need to recreate the same variables as in the model.
tf.reset_default_graph()

# Build graph
graph = tf.Graph()
learning_rate = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.image_size, FLAGS.image_size, 3])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
prediction, clstm_3 = clstm(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=prediction, labels=y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)


# Now load the checkpoint variable values

validation_dataset = create_dataset(FLAGS.val_data)

# Re-initializable iterator
iterator = tf.data.Iterator.from_structure(
    validation_dataset.output_types, validation_dataset.output_shapes)
next_element = iterator.get_next()

validation_init_op = iterator.make_initializer(validation_dataset,
                                               name='val_init_op')


if not os.path.exists(FLAGS.output_folder):
    os.makedirs(FLAGS.output_folder)


STEPS_VAL = int(FLAGS.nb_val_samples/FLAGS.batch_size)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/workspace/checkpoints/3lyr_32_mom_wholeseq_bs8")
    
    sess.run(validation_init_op)
    y_true = []
    y_hat = []
    guesses5 = []

    for i in range(STEPS_VAL):
        sequence, label = sess.run(next_element)
        print(i, end="\r")

        preds = sess.run(prediction, feed_dict={x: sequence, y: label})
        top5 = get_top_k(preds[0], 5).tolist()

        if(np.argmax(label) in top5):
            guesses5.append(np.argmax(label, axis=1))
            print('label {} was in top5 guesses'.format(np.argmax(label)))
        else:
            guesses5.append(np.argmax(preds, axis=1))

        y_true.append(np.argmax(label, axis=1))
        y_hat.append(np.argmax(preds, axis=1))


np.save(FLAGS.output_folder + 'y_true_3_32.npy', y_true)
np.save(FLAGS.output_folder + 'y_hat_3_32.npy', y_hat)
np.save(FLAGS.output_folder + 'y_hat_top5_3_32.npy', guesses5)

