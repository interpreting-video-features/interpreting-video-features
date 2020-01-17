import tensorflow as tf

tf.enable_eager_execution()

NUM_CLASSES = 6
PAD_LENGTH = 400

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_width',
    160,
    """Image width.""")
tf.app.flags.DEFINE_integer('image_height',
    120,
    """Image height.""")

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
    frame_inds = tf.range(parsed_features['nb_frames'])

    # Decode the image strings
    images = tf.map_fn(lambda i: \
                       tf.cast(tf.image.decode_jpeg(
                        parsed_features["frames"].values[i]),
                       dtype=tf.int32),
                       frame_inds,
                       dtype=tf.int32)

    images = tf.cast(images, tf.float32)
    label = tf.one_hot(parsed_features['label'], NUM_CLASSES)
    label = tf.cast(label, tf.int32)
    
    images.set_shape([PAD_LENGTH, FLAGS.image_height, FLAGS.image_width, 3])
    label.set_shape([NUM_CLASSES])

    return images, label


filenames = ['/data/tfrecords/kth_subject_1.tfrecords']
raw_dataset = tf.data.TFRecordDataset(filenames)

parsed_dataset = raw_dataset.map(parse_fn)

for parsed_record in parsed_dataset.take(1):
    print(repr(parsed_record))

