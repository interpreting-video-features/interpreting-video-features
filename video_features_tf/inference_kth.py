import tensorflow as tf
import pandas as pd
import numpy as np
from models import clstm
import ast
import os

CLIP_LENGTH = 32
NUM_CLASSES = 6


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(tf.flags.FLAGS)

NUM_CLASSES = 6
TOP_X = 3  # top x result to compute.
lr = 0.001

tf.app.flags.DEFINE_integer('nb_epochs',
    10,
    """Number of epochs to train.""")
tf.app.flags.DEFINE_integer('seq_length',
    32,
    """Length of video clips (nb of frames).""")
tf.app.flags.DEFINE_string('layers',
    '[32,32]',
    """Number of hidden units per CLSTM-layer, given as a list.
       For example [64,32,16]. The number of layers will be implicit
       from len(list).""")
tf.app.flags.DEFINE_string('return_sequences',
    '[True,True]',
    """Whether to return the full sequence or only the
       last element at every hidden layer for CLSTM.""")
tf.app.flags.DEFINE_string('only_last_element_for_fc',
    'no',
    """Whether to give only the last element from the last CLSTM
        layer to the FC layer, even if return_sequences=True for that layer.""")
tf.app.flags.DEFINE_string('pooling_method',
    'max',
    """avg|max""")
tf.app.flags.DEFINE_integer('kernel_size_1',
    3,
    """First size of convolutional kernel in clstm-unit.""")
tf.app.flags.DEFINE_integer('kernel_size_2',
    5,
    """Second size of convolutional kernel in clstm-unit.""")
tf.app.flags.DEFINE_integer('image_width',
    160,
    """Image width.""")
tf.app.flags.DEFINE_integer('image_height',
    120,
    """Image height.""")
tf.app.flags.DEFINE_integer('shuffle_buffer',
    2500,
    """Shuffle buffer.""")
tf.app.flags.DEFINE_integer('batch_size',
    24,
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
    0.9,
    """Momentum for the gradient descent.""")
tf.app.flags.DEFINE_float('weight_decay',
    0.00001,
    """Decoupled weight decay for momentum optimizer).""")
tf.app.flags.DEFINE_integer('nb_parallel_calls',
    16,
    """Number of parallel core calls when preparing a batch.""")
tf.app.flags.DEFINE_integer('lr_decay_patience',
    2,
    """Number of epochs without improvement to wait before decreased learning rate.""")
tf.app.flags.DEFINE_string('shuffle_data',
    'yes',
    """Whether to shuffle the training data. See SHUFFLE_BUFFER param.""")
tf.app.flags.DEFINE_string('train_subjects',
    '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]',
    """Subjects to train on.""")
tf.app.flags.DEFINE_string('val_subjects',
    '[17,18,19,20,21,22,23,24,25]',
    """Subjects to validate on.""")
tf.app.flags.DEFINE_string('checkpoint_name',
    'model.ckpt',
    """To go in checkpoints/model.ckpt""")
tf.app.flags.DEFINE_string('tfrecords_folder',
    '/data/kth_dataset/tfrecords_sample32/',
    """Directory containing the subject .tfrecords files.""")
tf.app.flags.DEFINE_string('test_run',
    'no',
    """Whether to only do a test run with few steps. 'yes' if so.""")
tf.app.flags.DEFINE_string('model',
    'clstm',
    """Which model to run with. cnn_3d | clstm""")
tf.app.flags.DEFINE_string('optimizer',
    'adadelta',
    """Which optimizer to run with. adadelta | momentum | momentum_decoupled""")
tf.app.flags.DEFINE_float('learning_rate_start',
    0.001,
    """Learning rate to start with.""")
tf.app.flags.DEFINE_float('learning_rate_end',
    0.001,
    """Minimum learning rate after decay.""")
tf.app.flags.DEFINE_float('kernel_regularizer',
    0.01,
    """Kernel regularizer for the ConvLSTM2D layer.""")
tf.app.flags.DEFINE_string('output_folder',
    None,
    """Where to save predictions as .npy files""")

print(tf.app.flags.FLAGS.flag_values_dict())

FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 6
PAD_LENGTH = 32

subjects_clips_df = pd.read_csv('/data/subjects_clips.csv')

train_subjects = ast.literal_eval(FLAGS.train_subjects)
val_subjects = ast.literal_eval(FLAGS.val_subjects)

print('Subjects to train on: ', train_subjects)
print('Subjects to test on: ', val_subjects)

nb_train_samples = 0
nb_val_samples = 0
train_tfrecords = []
val_tfrecords = []

for s in train_subjects:
    nb_clips = subjects_clips_df.at[s-1, 'nb_clips']  # Subject 1 is at index 0 etc.
    nb_train_samples += nb_clips
    train_tfrecords.append(FLAGS.tfrecords_folder + 'kth_subject_' + str(s) + '.tfrecords')

for s in val_subjects:
    nb_clips = subjects_clips_df.at[s-1, 'nb_clips']  # Subject 1 is at index 0 etc.
    nb_val_samples += nb_clips
    val_tfrecords.append(FLAGS.tfrecords_folder + 'kth_subject_' + str(s) + '.tfrecords')

if FLAGS.test_run == 'yes':
    STEPS_TRAIN = 5
    STEPS_VAL = 5
    FLAGS.shuffle_data = 'no'
    model_dir = 'ckpts_test/'

else:
    STEPS_TRAIN = int(nb_train_samples / FLAGS.batch_size)
    STEPS_VAL = int(nb_val_samples / FLAGS.batch_size)
    model_dir = 'checkpoints/'


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
    # TEST REVERSE PERFORMANCE
    # images = tf.reverse(tensor=images, axis=[0])
    # import pdb; pdb.set_trace()
    label = tf.one_hot(parsed_features['label'], NUM_CLASSES)
    label = tf.cast(label, tf.int32)

    return images, label


def create_dataset(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
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
x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.image_height, FLAGS.image_width, 3])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
prediction, clstm_3 = clstm.clstm(x, bn=False, is_training=False, num_classes=NUM_CLASSES)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=prediction, labels=y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)


# Now load the checkpoint variable values

validation_dataset = create_dataset(val_tfrecords)

# Re-initializable iterator
iterator = tf.data.Iterator.from_structure(
    validation_dataset.output_types, validation_dataset.output_shapes)
next_element = iterator.get_next()

validation_init_op = iterator.make_initializer(validation_dataset,
                                               name='val_init_op')


if not os.path.exists(FLAGS.output_folder):
    os.makedirs(FLAGS.output_folder)


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/workspace/checkpoints/" + FLAGS.checkpoint_name)
    
    sess.run(validation_init_op)
    y_true = []
    y_hat = []
    guesses5 = []

    for i in range(STEPS_VAL):
        sequence, batch_label = sess.run(next_element)
        print(i, end="\r")

        batch_preds = sess.run(prediction, feed_dict={x: sequence, y: batch_label})

        for b in range(FLAGS.batch_size):
            preds = batch_preds[b]
            label = batch_label[b]

            top5 = get_top_k(preds, TOP_X).tolist()

            if np.argmax(label) in top5:
                guesses5.append(np.argmax(label))
                print('label {} was in top {} guesses'.format(np.argmax(label), TOP_X))
            else:
                guesses5.append(np.argmax(preds))

            y_true.append(np.argmax(label))
            y_hat.append(np.argmax(preds))


np.save(FLAGS.output_folder + 'y_true_3_32.npy', y_true)
np.save(FLAGS.output_folder + 'y_hat_3_32.npy', y_hat)
np.save(FLAGS.output_folder + 'y_hat_top5_3_32.npy', guesses5)

