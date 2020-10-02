import tensorflow as tf
import pandas as pd
import numpy as np
from models import clstm
import ast
import os

FLAGS = tf.app.flags.FLAGS
CLIP_LENGTH = 32
TOP_X = 3  # top x result to compute.

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
    label = tf.one_hot(parsed_features['label'], FLAGS.nb_classes)
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
y = tf.placeholder(tf.float32, [None, FLAGS.nb_classes])
prediction, clstm_3 = clstm.clstm(x, bn=False, is_training=False, num_classes=FLAGS.nb_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=prediction, labels=y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate_start)


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

