import tensorflow as tf
from skimage.transform import resize
from keras.utils import np_utils
import gradcam as gc
import pandas as pd
import numpy as np
import mask
import viz
import ast
import cv2
import os

NUM_CLASSES = 174

NUM_CLASSES = 174
lr = 0.001

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
tf.app.flags.DEFINE_float('lambda_1',
    0.01,
    """First weight in find mask loss function.""")
tf.app.flags.DEFINE_float('lambda_2',
    0.02,
    """Second weight in find mask loss function.""")
tf.app.flags.DEFINE_integer('nb_iterations_graddescent',
    100,
    """Number of iterations when running gradient descent for the find mask method.""")
tf.app.flags.DEFINE_string('clip_selection',
    None,
    """A .csv-file containing the clips to run on.""")
tf.app.flags.DEFINE_string('focus_type',
    None,
    """guessed|correct (whether to base the gradcam on the classified or correct class""")
tf.app.flags.DEFINE_string('normalization_mode',
    None,
    """frame|sequence (whether to normalize the gradcam to the seq or frame.""")
tf.app.flags.DEFINE_string('temporal_mask_type',
    None,
    """freeze|reverse method of masking in temporal mask.""")

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
            
    with tf.name_scope('fully_con'):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=NUM_CLASSES)

    return x, clstm_3


def parse_fn(proto):

    # Define the tfrecord again. The sequence was saved as a string.
    keys_to_features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'nb_frames': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'video_id': tf.FixedLenFeature([], tf.string),
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
    label = tf.one_hot(parsed_features['label'], NUM_CLASSES)
    label = tf.cast(label, tf.int32)

    video_ID = parsed_features['video_id']

    return images, label, video_ID

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


def main(argv):

    df = pd.read_csv(FLAGS.clip_selection)  # DataFrame containing the clips to run on.

    # First we need to recreate the same variables as in the model.
    tf.reset_default_graph()
    seq_shape = (FLAGS.batch_size, FLAGS.seq_length, FLAGS.image_size, FLAGS.image_size, 3)
    seq_zeros = np.zeros(seq_shape)
    
    # Build graph
    graph = tf.Graph()

    # Graph for perturb_sequence(seq, mask, perb_type) method
    # Create variable to save original input sequence
    with tf.variable_scope('original_input'):
        original_input_plhdr = tf.placeholder(tf.float32, seq_shape)
        original_input_var = tf.get_variable('original_input',
                                   seq_shape,
                                   dtype=tf.float32,
                                   trainable=False)
        original_input_assign = original_input_var.assign(original_input_plhdr)

    x = tf.placeholder(tf.float32, seq_shape)

    with tf.variable_scope('mask'):
        # Create variable for the temporal mask
        mask_plhdr = tf.placeholder(tf.float32, [FLAGS.seq_length])
        mask_var = tf.get_variable('input_mask',
                                   [FLAGS.seq_length],
                                   dtype=tf.float32,
                                   trainable=True)
        mask_assign = tf.assign(mask_var, mask_plhdr)
        mask_clip = tf.nn.sigmoid(mask_var)

    with tf.variable_scope('perturb'):

        frame_inds = tf.placeholder(tf.int32, shape=(None,), name='frame_inds')

        # if FLAGS.temporal_mask_type == 'freeze':
        
        def recurrence(last_value, current_elem):
            update_tensor = (1-mask_clip[current_elem])*original_input_var[:,current_elem,:,:,:] + \
                            mask_clip[current_elem]*last_value
            return update_tensor
        
        perturb_op = tf.scan(fn=recurrence,
                             elems=frame_inds,
                             initializer=original_input_var[:,0,:,:,:])
        perturb_op = tf.reshape(perturb_op, seq_shape)


    y = tf.placeholder(tf.float32, [FLAGS.batch_size, NUM_CLASSES])
    logits, clstm_3 = clstm(perturb_op)
    after_softmax = tf.nn.softmax(logits)
    
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
    # Settings for temporal mask method
    N = FLAGS.nb_iterations_graddescent
    maskType='gradient'
    verbose=True
    maxMaskLength=None
    do_gradcam=True
    run_temp_mask=True
    ita = 1
            
    variables_to_restore = {}
    for variable in tf.global_variables():
        if variable.name.startswith('mask'):
            continue
        elif variable.name.startswith('original_input'):
            continue
        else:
            # Variables need to be renamed to match with the checkpoint.
            variables_to_restore[variable.name.replace(':0','')] = variable
    
    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=variables_to_restore)  # All but the input which is a variable
        saver.restore(sess, "/workspace/checkpoints/3lyr_32_mom_wholeseq_bs8")
        
        sess.run(validation_init_op)

        l1loss = FLAGS.lambda_1*tf.reduce_sum(tf.abs(mask_clip))
        tvnormLoss= FLAGS.lambda_2*calc_TVNorm(mask_clip, p=3, q=3)
        if FLAGS.focus_type == 'correct':
            label_index = tf.reshape(tf.argmax(y, axis=1), [])
        if FLAGS.focus_type == 'guessed':
            label_index = tf.reshape(tf.argmax(logits, axis=1), [])
        class_loss = after_softmax[:, label_index]
        # Cast as same type as l1 and TV.
        class_loss = tf.cast(class_loss, tf.float32)

        loss_function = l1loss + tvnormLoss + class_loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_start)
        train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mask')

        with tf.variable_scope('minimize'):
            training_op = optimizer.minimize(loss_function, var_list=train_var)
    
        sess.run(tf.variables_initializer(optimizer.variables()))
        
        for i in range(STEPS_VAL):

            input_var, label, video_ID = sess.run(next_element)

            # video_ID is returned as an array of a bytes object
            video_ID = video_ID[0].decode("utf-8")  # like so: array([b'74225'], dtype=object)

            # only look at cases where the network was correct, and is of a certain class (if class of interest 'classoI' was given)
            current_class = np.argmax(label)
            print(current_class)

            current_class = str(np.argmax(label))

            if current_class in list(df.keys()) and int(video_ID) in [derp for derp in df[current_class]]:

                print("found clip of interest ", current_class, video_ID)

                preds = sess.run(logits, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                 original_input_var: input_var,
                                 frame_inds: range(FLAGS.seq_length)})
                print('np argmax preds', np.argmax(preds))

                masks = []
                
                #eta is for breaking out of the grad desc early if it hasn't improved
                eta = 0.00001
                
                have_output=False

                if preds[:, int(current_class)] < 0.1:
                    print('the guess for the correct class was less than 0.1')
                    continue
                
                if not have_output:
                    output = sess.run(after_softmax, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                     original_input_var: input_var,
                                     frame_inds: range(FLAGS.seq_length)})
                    have_output=True
                
                if(run_temp_mask):
                    if (maskType == 'gradient'):
                        start_mask = mask.init_mask(input_var, mask_var, original_input_var,
                                              frame_inds, after_softmax, sess,
                                              label, thresh=0.9,
                                              mode="central", mask_pert_type=FLAGS.temporal_mask_type)
                        # Initialize mask variable
                        sess.run(mask_assign, {mask_plhdr: start_mask})
                        sess.run(original_input_assign,
                                 {original_input_plhdr: input_var})

                        oldLoss = 999999
                        for nidx in range(N):

                            if(nidx%10==0):
                                print("on nidx: ", nidx)
                                print("mask_clipped is: ", sess.run(mask_clip))
                            
                            _, loss_value, \
                            l1value, tvvalue, classlossvalue = sess.run([training_op,
                                                                        loss_function,
                                                                        l1loss,
                                                                        tvnormLoss,
                                                                        class_loss],
                                                                        feed_dict={y: label,
                                                                                   frame_inds: range(FLAGS.seq_length),
                                                                                   original_input_var: input_var})

                            print("LOSS: {}, l1loss: {}, tv: {}, class: {}".format(loss_value,
                                                                                   l1value,
                                                                                   tvvalue,
                                                                                   classlossvalue))
                            if(abs(oldLoss-loss_value)<eta):
                                break;

                        
                        time_mask = sess.run(mask_clip)
                        save_path = os.path.join("cam_saved_images",
                                                 FLAGS.output_folder,
                                                 str(np.argmax(label)),
                                                 video_ID + "g_" + \
                                                 str(np.argmax(output)) + \
                                                 "_cs%5.4f"%output[:,np.argmax(label)] + \
                                                 "gs%5.4f"%output[:,np.argmax(output)],
                                                 "combined")
                        
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        
                        f = open(save_path+"/ClassScoreFreezecase"+video_ID+".txt","w+")
                        f.write(str(classlossvalue))
                        f.close()

                        if FLAGS.temporal_mask_type == 'reverse':

                            perturbed_sequence = mask.perturb_sequence(input_var, time_mask, perb_type='reverse')
                            
                            class_loss_rev = sess.run(class_loss, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                                                             original_input_var: perturbed_sequence,
                                                                             frame_inds: range(FLAGS.seq_length)})
                            f = open(save_path+"/ClassScoreReversecase" + video_ID + ".txt","w+")
                            f.write(str(class_loss_rev))
                            f.close()

                    if(verbose):
                        print("resulting mask is: ", sess.run(mask_clip))

                if(do_gradcam):  

                    if(FLAGS.focus_type=="guessed"):
                        target_index=np.argmax(output)
                    if(FLAGS.focus_type=="correct"):
                        target_index=np.argmax(label)

                    gradcam = get_gradcam(sess, logits, clstm_3, y, original_input_var, mask_var, frame_inds,
                                          input_var, label, target_index, FLAGS.image_size, FLAGS.image_size)

                    '''beginning of gradcam write to disk'''
                    

                    os.makedirs(save_path, exist_ok=True)
                    
                if(do_gradcam and run_temp_mask):
                    viz.create_image_arrays(input_var, gradcam, time_mask,
                                        save_path, video_ID, 'freeze',
                                        FLAGS.image_size, FLAGS.image_size)

                    if FLAGS.temporal_mask_type == 'reverse':
                        # Also create the image arrays for the reverse operation.
                        viz.create_image_arrays(input_var, gradcam, time_mask,
                                            save_path, video_ID, 'reverse',
                                            FLAGS.image_size, FLAGS.image_size)
                
                if(run_temp_mask):
                    viz.visualize_results(input_var,
                                      mask.perturb_sequence(input_var,
                                                      time_mask, perb_type='reverse'),
                                      time_mask,
                                      root_dir=save_path,
                                      case=video_ID, mark_imgs=True,
                                      iter_test=False)
    

if __name__ == '__main__':
    tf.app.run()
            
