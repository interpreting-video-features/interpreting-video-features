import tensorflow as tf
import numpy as np
import os

from models import clstm
from configs import config_train_smth_clstm

FLAGS = tf.app.flags.FLAGS

# Where to save predictions as .npy files
output_dir = os.path.join(FLAGS.workspace_dir, 'output/')


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
                       parsed_features['frames'].values[i]),
                       dtype=tf.int32),
                       frame_inds,
                       dtype=tf.int32)

    images = tf.cast(images, tf.float32)
    # UNCOMMENT TO TEST REVERSE PERFORMANCE
    # images = tf.reverse(tensor=images, axis=[0])
    label = tf.one_hot(parsed_features['label'], FLAGS.nb_classes)
    label = tf.cast(label, tf.int32)

    return images, label


def create_dataset(filepath):
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

# First we need to recreate the same variables as in the model.
tf.reset_default_graph()

# Build graph
graph = tf.Graph()
learning_rate = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [None,
                                FLAGS.seq_length,
                                FLAGS.image_size,
                                FLAGS.image_size,
                                3])
y = tf.placeholder(tf.float32, [None, FLAGS.nb_classes])
prediction, clstm_3 = clstm.clstm(x, bn=True, num_classes=FLAGS.nb_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=prediction, labels=y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate_start)


# Now load the checkpoint variable values

validation_dataset = create_dataset(FLAGS.val_data)

# Re-initializable iterator
iterator = tf.data.Iterator.from_structure(
    validation_dataset.output_types, validation_dataset.output_shapes)
next_element = iterator.get_next()

validation_init_op = iterator.make_initializer(validation_dataset,
                                               name='val_init_op')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


STEPS_VAL = int(FLAGS.nb_val_samples/FLAGS.batch_size)

with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(FLAGS.workspace_dir,
                                   'checkpoints',
                                   FLAGS.checkpoint_name)
    saver.restore(sess, checkpoint_path)
    
    sess.run(validation_init_op)
    y_true = []
    y_hat = []
    guesses5 = []

    for i in range(STEPS_VAL):
        sequence, label = sess.run(next_element)
        print('\r{}'.format(i))

        preds = sess.run(prediction, feed_dict={x: sequence, y: label})
        top5 = get_top_k(preds[0], 5).tolist()

        if np.argmax(label) in top5:
            guesses5.append(np.argmax(label, axis=1))
            print('label {} was in top5 guesses'.format(np.argmax(label)))
        else:
            guesses5.append(np.argmax(preds, axis=1))

        y_true.append(np.argmax(label, axis=1))
        y_hat.append(np.argmax(preds, axis=1))


np.save(output_dir + 'y_true_3_32.npy', y_true)
np.save(output_dir + 'y_hat_3_32.npy', y_hat)
np.save(output_dir + 'y_hat_top5_3_32.npy', guesses5)

