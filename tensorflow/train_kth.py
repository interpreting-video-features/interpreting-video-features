from comet_ml import Experiment

from tensorflow.python import keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import ast
import os

from models import clstm, cnn_3d, i3d

experiment = Experiment(api_key="xAURnaQRjUuVQO68jQZEUEDgj",
                        project_name="kth-actions", workspace="sofiabroome")

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_epochs',
    300,
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
    'yes',
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
    0.5,
    """The rate at which to perform dropout (how much to drop).""")
tf.app.flags.DEFINE_float('momentum',
    0.2,
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
    '/data/tfrecords/',
    """Directory containing the subject .tfrecords files.""")
tf.app.flags.DEFINE_string('test_run',
    'no',
    """Whether to only do a test run with few steps. 'yes' if so.""")
tf.app.flags.DEFINE_string('model',
    'clstm',
    """Which model to run with. cnn_3d | clstm""")
tf.app.flags.DEFINE_string('optimizer',
    'momentum',
    """Which optimizer to run with. adadelta | momentum | momentum_decoupled""")
tf.app.flags.DEFINE_float('learning_rate_start',
    0.001,
    """Learning rate to start with.""")
tf.app.flags.DEFINE_float('learning_rate_end',
    0.00000001,
    """Minimum learning rate after decay.""")
tf.app.flags.DEFINE_float('kernel_regularizer',
    0.01,
    """Kernel regularizer for the ConvLSTM2D layer.""")

NUM_CLASSES = 6

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

# Log parameters to Comet.ml!
params = {'STEPS_VAL': STEPS_VAL,
          'STEPS_TRAIN': STEPS_TRAIN,
          'FLAGS.layers': FLAGS.layers,
          'FLAGS.checkpoint_name': FLAGS.checkpoint_name,
          'FLAGS.model': FLAGS.model,
          'FLAGS.learning_rate_start': FLAGS.learning_rate_start,
          'FLAGS.learning_rate_end': FLAGS.learning_rate_end}

experiment.log_parameters(params)


def i3d_model(rgb_input, is_training):
    rgb_model = i3d.InceptionI3d(
        NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
        rgb_input, is_training=is_training, dropout_keep_prob=(1-FLAGS.dropout_rate))
    return rgb_logits


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
    
    images.set_shape([FLAGS.seq_length, FLAGS.image_height, FLAGS.image_width, 3])
    label.set_shape([NUM_CLASSES])

    return images, label

  
def create_dataset(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    if FLAGS.shuffle_data == 'yes':
        print('Shuffling the training data...')
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
    dataset = dataset.map(map_func=parse_fn,
                          num_parallel_calls=FLAGS.nb_parallel_calls)
    dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
                                   padded_shapes=([FLAGS.seq_length,
                                                  FLAGS.image_height,
                                                  FLAGS.image_width, 3],
                                                  [NUM_CLASSES]))
    dataset = dataset.prefetch(FLAGS.batch_size)
    return dataset


def main(argv):
    print(tf.app.flags.FLAGS.flag_values_dict())

    print('STEPS_TRAIN: ', STEPS_TRAIN)
    print('STEPS_VAL: ', STEPS_VAL)
    
    # Define tensor for updating global step
    global_step = tf.train.get_or_create_global_step()
    
    training_dataset = create_dataset(train_tfrecords)
    validation_dataset = create_dataset(val_tfrecords)
    
    # Graph
    learning_rate = tf.placeholder(tf.float32, [])
    x = tf.placeholder(tf.float32,
                      [None, FLAGS.seq_length, FLAGS.image_height, FLAGS.image_width, 3])
    y = tf.placeholder(tf.int32, [None, NUM_CLASSES])
    is_training = tf.placeholder(tf.bool, (), 'is_training')
    
    # Get logits from the chosen model.
    if FLAGS.model == 'cnn_3d':    
        prediction = cnn_3d.cnn_3d(x) 
    if FLAGS.model == 'clstm':    
        prediction, _ = clstm.clstm(x, bn=False,
                                 is_training=is_training,
                                 num_classes=NUM_CLASSES)
    if FLAGS.model == 'clstm_bn':    
        prediction, _ = clstm.clstm(x, bn=True,
                                 is_training=is_training,
                                 num_classes=NUM_CLASSES)
    if FLAGS.model == 'clstm_gap':    
        prediction = clstm.clstm_gap(x, bn=False,
                               is_training=is_training,
                               num_classes=NUM_CLASSES)
    if FLAGS.model.startswith('i3d'):    
        prediction = i3d_model(x, is_training=is_training)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))
    
    if FLAGS.optimizer == 'momentum_decoupled':
        print('Momentum optimizer with decoupled weight decay')
        lr = FLAGS.learning_rate_start
        lr_end = FLAGS.learning_rate_end
        # Create an extended MomentumOptimizer object.
        OptExt = tf.contrib.opt.extend_with_decoupled_weight_decay( \
            tf.train.MomentumOptimizer)
        optimizer = OptExt(weight_decay=FLAGS.weight_decay,
            learning_rate=lr,
            momentum=FLAGS.momentum)

    if FLAGS.optimizer == 'momentum':
        print('Momentum opt')
        lr = FLAGS.learning_rate_start
        lr_end = FLAGS.learning_rate_end

        if FLAGS.weight_decay > 0.0:
            optimizer = tf.contrib.opt.MomentumWOptimizer( \
                learning_rate=lr,
                weight_decay=FLAGS.weight_decay,
                momentum=FLAGS.momentum)
        else:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=lr,
                momentum=FLAGS.momentum)

    if FLAGS.optimizer == 'sgd':
        print('SGD opt')
        lr = FLAGS.learning_rate_start
        lr_end = FLAGS.learning_rate_end
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr)

    if FLAGS.optimizer == 'adadelta':
        print('Adadelta opt')
        lr = FLAGS.learning_rate_start
        lr_end = FLAGS.learning_rate_end
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr) 

    if FLAGS.optimizer == 'adam':
        print('Adam opt')
        lr = FLAGS.learning_rate_start
        lr_end = FLAGS.learning_rate_end
        optimizer = tf.train.AdamOptimizer(learning_rate=lr) 

    training_op = optimizer.minimize(loss,
        global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Re-initializable iterator
    iterator = tf.data.Iterator.from_structure(
        training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset,
                                                   name='train_init_op')
    validation_init_op = iterator.make_initializer(validation_dataset,
                                                   name='val_init_op')
    
    step_times = []
    best_val_acc = 0
    patient_epochs = 0

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Initialize all variables and start training.
        tf.global_variables_initializer().run(session=sess)
    
        # Log model graph
        experiment.set_model_graph(sess.graph)
    
        ckpt_path = model_dir + FLAGS.checkpoint_name

        # Check if checkpoint exists for this model, restore if so.
        if os.path.isfile(ckpt_path + '.index'):
            print('Restoring checkpoint from: ' + ckpt_path + '...')
            saver.restore(sess, ckpt_path)
            # Get the global step reached by the already trained model.
            step = tf.train.global_step(sess, global_step)
            # Compute which epoch that corresponds to.
            start_ep = int(step/STEPS_TRAIN)

        elif FLAGS.model == 'i3d_pretrained':
            print('Restoring I3D checkpoint...')
            rgb_variable_map = {}
            for variable in tf.global_variables():
                print(variable)
                # Exclude logit layer since we are finetuning on a new set of classes.
                if not variable.name.startswith('inception_i3d/Logits'):
                    # Variables need to be renamed to match with the checkpoint.
                    rgb_variable_map['RGB/'+ variable.name.replace(':0','')] = variable
            del rgb_variable_map['RGB/global_step']
            if FLAGS.optimizer == 'adam':
                del rgb_variable_map['RGB/beta1_power']
                del rgb_variable_map['RGB/beta2_power']
                elts_to_delete = []
                for k in rgb_variable_map.keys():
                    if 'Adam' in k:
                        print(k)
                        elts_to_delete.append(k)
                for elt in elts_to_delete:
                    del rgb_variable_map[elt]
                
            # Add ops to save and restore for the chosen variables.
            saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
            saver.restore(sess, 'checkpoints/rgb_imagenet/model.ckpt')
            # saver.restore(sess, 'checkpoints/rgb_scratch/model.ckpt')
            start_ep = 0
        else:
            print('Training a new model from scratch...')
            start_ep = 0

        # Start training.
        for ep in range(start_ep, FLAGS.nb_epochs):

            print('\n')
            print('Epoch: ', ep+1)
            start_epoch = time.time()

            # Initialize iterators in the beginning of each epoch
            sess.run(training_init_op)

            for st in range(STEPS_TRAIN): 
                # Get the global step.
                step = tf.train.global_step(sess, global_step)
                try:
                    s = time.time()
                    sequence, label = sess.run(next_element)
                    if step % 1 == 0:
                        _, acc, train_loss = sess.run([training_op, accuracy, loss],
                                           feed_dict={x: sequence,
                                                      y: label,
                                                      is_training: True,
                                                      learning_rate: lr})
                        experiment.log_metric("accuracy", acc, step=step)
                        experiment.log_metric("loss", train_loss, step=step)
                    else:
                        sess.run(training_op, feed_dict={x: sequence,
                                                         y: label,
                                                         is_training: True,
                                                         learning_rate: lr})
                    e = time.time()
                    t = e - s
                    step_times.append(t)
    
                except tf.errors.OutOfRangeError:
                    print('Inside OutOfRangeError...')
                    break
        
            end_epoch = time.time()
            time_epoch = end_epoch - start_epoch
            print('Time taken for training epoch: {:0.2f} s\n'.format(time_epoch))
    
            val_losses = []
            val_accs = []
            print('Testing on the validation set...')
            start_val = time.time()
            sess.run(validation_init_op)
            for _ in range(STEPS_VAL):
                sequence, label = sess.run(next_element)
                val_acc, val_loss = sess.run([accuracy, loss],
                                             feed_dict={x: sequence,
                                                        y: label,
                                                        is_training: False,
                                                        learning_rate: lr})
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                experiment.log_metric("val. acc.", val_acc)
                experiment.log_metric("val. loss.", val_loss)
                experiment.log_metric("learning rate", lr)

            end_val = time.time()
            time_val = end_val - start_val
            print('Time taken for validation inference: {:0.2f} s\n'.format(time_val))
            val_losses = np.asarray(val_losses)
            mean_val_loss = np.mean(val_losses)
            experiment.log_metric("mean val. loss.", mean_val_loss)
            print('Average validation loss: {:0.6f}'.format(mean_val_loss))

            val_accs = np.asarray(val_accs)
            mean_val_acc = np.mean(val_accs)
            experiment.log_metric("mean val. acc.", mean_val_acc)
            print('Average validation acc: {:0.6f}'.format(mean_val_acc))
    
            if (mean_val_acc - best_val_acc) < 0.0001:

                print('The validation accuracy did not improve.')
                print('Best so far: {:0.6f}'.format(best_val_acc))
                print('Mean val acc this ep: {:0.6f}\n'.format(mean_val_acc))
                print('Incrementing patient epoch count...\n')
                patient_epochs += 1

                if patient_epochs == FLAGS.lr_decay_patience:
                    print('{} epochs went by '.format(FLAGS.lr_decay_patience) \
                          + 'without mean_val_acc improvement...')
                    if lr >= 2*lr_end:
                        print('Reducing learning rate by a factor of 0.5.\n')
                        lr = 0.5 * lr
                    patient_epochs = 0  # Reset patient epochs count
            else: # Update the best acc for the next epoch if it has improved
                print('The validation acc improved.')
                best_val_acc = mean_val_acc
                patient_epochs = 0  # Reset patient epochs count
        
                # Save the variables to disk.
                print('Saving checkpoint!')
                cwd = os.getcwd()
                path = os.path.join(cwd, model_dir)
                print('path: ', path)
                save_path = saver.save(sess, path + FLAGS.checkpoint_name)
                print("Model saved in path: %s" % save_path) 

    # Average step time across training
    step_times = np.asarray(step_times)
    print('Training step times:\n')
    print(step_times, '\n')
    print('Average duration for one step of training: {:0.4f} s'.format(np.mean(step_times)))
    print('\n')

if __name__ == '__main__':
    tf.app.run()

