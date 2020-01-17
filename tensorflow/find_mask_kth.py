import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report
from skimage.transform import resize
from keras.utils import np_utils
from PIL import Image
import pandas as pd
import numpy as np
import clstm
import math
import ast
import cv2
import os

NUM_CLASSES = 6
MASK_THRESHOLD = 0.1
PATH_TO_FRAMES = None


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
    'no',
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
tf.app.flags.DEFINE_string('output_folder',
    None,
    """Where to save gradcam blends""")

print(tf.app.flags.FLAGS.flag_values_dict())

FLAGS = tf.app.flags.FLAGS

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


def get_paths_for_sequence(subject, video_id):
    stem = PATH_TO_FRAMES
    coi = str(subject) + '_' + video_id + '_1'

    for subdir, dirs, files in sorted(os.walk(stem)):
        for file in files:
            path = os.path.join(subdir, file)
            if path.endswith('.txt'):
                with open(path,"r") as fl:
                    tag = fl.readline()
                if coi == tag:
                    print(path)
                    print("COI: {}".format(coi))
                    clip_dir = path[:-9]
                    break
    frame_paths = []
    for i in range(FLAGS.seq_length):
        counter_format = "%02d" % (i + 1)
        frame_path = clip_dir + 'frame' + counter_format + '.jpg'
        frame_paths.append(frame_path)
    return frame_paths

        
def read_images_and_return_list(paths):
    list_to_return = []
    for p in paths:
        img = process_image(p, (FLAGS.image_height, FLAGS.image_width, 3))                                                                                                                                                                   
        img = img.reshape((1, 1, FLAGS.image_height, FLAGS.image_width, 3))
        list_to_return.append(img)
    return list_to_return


def process_image(image_path, target_shape):
    img = load_img(image_path, target_size=target_shape)
    return img_to_array(img).astype(np.float32)


def data_for_one_sequence_5D(paths, label):
    label_onehot = np_utils.to_categorical(label, num_classes=NUM_CLASSES)
    batch_label = label_onehot.reshape(FLAGS.batch_size, -1)
    batch_img = np.concatenate(read_images_and_return_list(paths), axis=1)
    batch_img = np.reshape(batch_img, (FLAGS.batch_size,
                                       FLAGS.seq_length,
                                       FLAGS.image_height,
                                       FLAGS.image_width,
                                       3))
    return batch_img, batch_label


def get_gradcam_blend(img, cam, cam_max):
    """
    img: np.ndarray
    cam: np.ndarray
    cam_max: float
    return: Image object"""
    cam = cam / cam_max # scale 0 to 1.0
    cam = resize(cam, (FLAGS.image_height, FLAGS.image_width), preserve_range=True)

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    bg = Image.fromarray((255*img).astype('uint8'))
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.4)

    return blend


def get_cam_after_relu(img, conv_output, conv_grad):
    weights = np.mean(conv_grad, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(conv_output.shape[0 : 2], dtype = np.float32) # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    return cam


def get_gradcam(sess, prediction, last_clstm_output,
                y, original_input_var, mask_var, frame_inds,
                sequence, label, target_index):
    
    prob = tf.keras.layers.Activation('softmax')(prediction)

    # Things needed for gradcam
    cost = (-1) * tf.reduce_sum(tf.multiply(y, tf.log(prob)), axis=1)
    
    # Elementwise multiplication between y and prediction, then reduce to scalar
    if target_index == np.argmax(label):
        y_c = tf.reduce_sum(tf.multiply(prediction, y), axis=1)
    else:
        # y_guessed = tf.one_hot(target_index, depth=1)
        y_guessed = tf.one_hot(tf.argmax(prob), depth=1)
        y_c = tf.reduce_sum(tf.multiply(prediction, y_guessed), axis=1)

    target_conv_layer = last_clstm_output

    # Compute gradients of class output wrt target conv layer
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
    
    # Obtain values for conv and grad tensors
    target_conv_layer_value, \
    target_conv_layer_grad_value = sess.run([target_conv_layer,
                                             target_conv_layer_grad],
                                             feed_dict={original_input_var: sequence,
                                                        y: label,
                                                        mask_var: np.zeros((FLAGS.seq_length)),
                                                        frame_inds: range(FLAGS.seq_length)})
    assert (target_conv_layer_grad_value.shape[1] == FLAGS.seq_length)

    # Get CAMs (class activation mappings) for all clips
    # and save max for normalization across clip.
    cam_max = 0
    gradcams = []
    for i in range(FLAGS.seq_length):
        frame = sequence[0,i,:]
        grad = target_conv_layer_grad_value[0,i,:]
        conv_output = target_conv_layer_value[0,i,:]
        # Prepare frame for gradcam
        img = frame.astype(float)
        img -= np.min(img)
        img /= img.max()
        cam = get_cam_after_relu(img,
                                 conv_output=conv_output,
                                 conv_grad=grad)
        gradcams.append(cam)
        if np.max(cam) > cam_max:
            cam_max = np.max(cam)
    gradcam_masks = [] 
    # Loop over frames again to blend them with the gradcams and save.
    for i in range(FLAGS.seq_length):
        frame = sequence[0,i,:]
        # Prepare frame for gradcam
        img = frame.astype(float)
        img -= np.min(img)
        img /= img.max()
    
        # NORMALIZE PER FRAME
        if FLAGS.normalization_mode == 'frame':
            gradcam_blend = get_gradcam_blend(img, gradcams[i], np.max(gradcams[i]))
        # NORMALIZE PER SEQUENCE 
        elif FLAGS.normalization_mode == 'sequence':
            gradcam_blend = get_gradcam_blend(img, gradcams[i], cam_max)
        else:
            print('Error. Need to provide normalization mode.')
        gradcam_masks.append(gradcam_blend)
    return gradcam_masks


def perturb_sequence(seq, mask, perb_type='freeze', snap_values=False):

    if(snap_values):
        for j in range(len(mask)):
            if(mask[j]>0.5):
                mask[j]=1
            else:
                mask[j]=0

    if (perb_type == 'freeze'):
        perbInput = np.zeros(seq.shape)
        for u in range(len(mask)):
            
            if(u==0):  # Set first frame to same as seq.
                perbInput[:,u,:,:,:] = seq[:,u,:,:,:]

            if(u!=0): #mask[u]>=0.5 and u!=0
                perbInput[:,u,:,:,:] = (1-mask[u])*seq[:,u,:,:,:] + \
                                        mask[u]*np.copy(perbInput)[:,u-1,:,:,:]
            
    if (perb_type == 'reverse'):
        #pytorch expects Batch,Channel, T, H, W
        perbInput = np.zeros(seq.shape)# seq.clone().detach()
        maskOnInds = np.where(mask>MASK_THRESHOLD)[0]  # np.where returns a tuple for some reason
        # if(len(maskOnInds)>0):
        #     maskOnInds = maskOnInds.squeeze(dim=1)
        maskOnInds = maskOnInds.tolist()
        
        subMasks = findSubMasksFromMask(mask)
        
        for y in range(len(mask)):
            perbInput[:,y,:,:,:]=seq[:,y,:,:,:]
                
        for maskOnInds in subMasks:
            #print("center should be, ", maskOnInds[(len(maskOnInds)//2)])
            #leave unmasked parts alone (as well as reverse middle point)
            if ((len(maskOnInds)//2 < len(maskOnInds)/2) and y==maskOnInds[(len(maskOnInds)//2)]):
                #print("hit center at ", y)
                perbInput[:,y,:,:,:]=seq[:,y,:,:,:]
            for u in range(int(len(maskOnInds)//2)):
                temp = seq[:,maskOnInds[u],:,:,:]
                perbInput[:,maskOnInds[u],:,:,:] = (1-mask[maskOnInds[u]])*seq[:,maskOnInds[u],:,:,:] + mask[maskOnInds[u]]*seq[:,maskOnInds[-(u+1)],:,:,:]
                perbInput[:,maskOnInds[-(u+1)],:,:,:] = (1-mask[maskOnInds[u]])*seq[:,maskOnInds[-(u+1)],:,:,:] + mask[maskOnInds[u]]*temp    
    #print("return type of pertb: ", perbInput.type())
    return perbInput


def findSubMasksFromMask(mask, thresh=MASK_THRESHOLD):
    subMasks = []
    currentSubMask = []
    currentlyInMask = False
    for j in range(len(mask)):
        #if we find a value above threshold but is first occurence, start appending to current submask
        if(mask[j]>thresh and not currentlyInMask):
            currentSubMask = []
            currentlyInMask=True
            currentSubMask.append(j)
        #if it's not current occurence, just keep appending
        elif(mask[j]>thresh and currentlyInMask):
            currentSubMask.append(j)
        #if below thresh, stop appending
        elif((mask[j]<=thresh and currentlyInMask)):
            subMasks.append(currentSubMask)
            currentlyInMask=False
            
        if(j==len(mask)-1 and currentlyInMask):
            subMasks.append(currentSubMask)
            currentlyInMask=False
    #print("submasks found: ", subMasks)
    return subMasks


def init_mask(seq, mask_var, original_input_var, frame_inds, after_softmax, sess, target,
             thresh=0.9, mode="central", mask_pert_type='freeze'):
    '''
    Initiaizes the first value of the mask where the gradient descent methods for finding
    the masks starts. Central finds the smallest centered mask which still reduces the class score by 
    at least 90% compared to a fully perturbing mask (whole mask on). Random just turns (on average) 70% of the 
    mask (Does not give very conclusive results so far). 
    '''
    if(mode=="central"):
        
        #first define the fully perturbed sequence
        fullPert = np.zeros(seq.shape)
        for i in range(seq.shape[1]):
            fullPert[:,i,:,:,:] = seq[:,0,:,:,:]
        
        #get the class score for the fully perturbed sequence
        full_pert_score = sess.run(after_softmax, feed_dict={mask_var: np.ones((FLAGS.seq_length)),
                                 original_input_var: seq,
                                 frame_inds: range(FLAGS.seq_length)})
        full_pert_score = full_pert_score[:,np.argmax(target)]
        
        orig_score = sess.run(after_softmax, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                 original_input_var: seq,
                                 frame_inds: range(FLAGS.seq_length)})
        orig_score = orig_score[:,np.argmax(target)]
            
        #reduce mask size while the loss ratio remains above 90%
        for i in range(1, seq.shape[1]//2):
            new_mask = np.ones(seq.shape[1])
            new_mask[:i]=0
            new_mask[-i:]=0

            central_score = sess.run(after_softmax, feed_dict={mask_var: new_mask,
                                     original_input_var: seq,
                                     frame_inds: range(FLAGS.seq_length)})
            central_score = central_score[:,np.argmax(target)]

            score_ratio=(orig_score-central_score)/(orig_score-full_pert_score)
            
            if(score_ratio < thresh):
                break
            
        mask=new_mask

        #modify the mask so that it is roughly 0 or 1 after sigmoid
        for j in range(len(mask)):
            if(mask[j]==0):
                mask[j]=-5
            elif(mask[j]==1):
                mask[j]=5
    
    elif(mode=="random"):
        #random init to 0 or 1, then modify for sigmoid
        mask = torch.cuda.FloatTensor(16).uniform_() > 0.7
        mask = mask.float()
        mask = mask - 0.5
        mask = mask*5
        
        #if mask were to be ALL 0's or 1's, perturb one a bit so that TV norm doesn't NaN
        if(torch.abs(mask.sum())==2.5*len(mask)):
            mask[8]+=0.1

    print("initial mask is: ", mask)
    return mask

      
def calc_TVNorm(mask, p=3, q=3):
    '''
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions in the mask.
    p=3 and q=3 are defaults from the paper.
    '''
    val = 0
    for u in range(1, FLAGS.seq_length-1):

        val += tf.abs(mask[u-1]-mask[u])**p
        val += tf.abs(mask[u+1]-mask[u])**p
    val = val**(1/p)
    val = val**q

    return val


def visualize_results_on_gradcam(gradcam_images, mask, root_dir,
                                 case="0", round_up_mask=True):
        
    #print("gradCamType: ", gradcam_images.type)
    try:
        mask=mask.detach().cpu()
    except:
        print("mask was already on cpu")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    dots = find_temp_mask_red_dots(FLAGS.image_width, FLAGS.image_height, mask, round_up_mask)
    
    dot_offset = FLAGS.image_width*2
    for i in range(len(mask)):
        for j,dot in enumerate(dots):

            if(i==j):
                intensity=255
            else:
                intensity=150

            gradcam_images[i][dot["y_start"]:,dot_offset+dot["x_start"]:dot_offset+dot["x_end"],:] = 0
            gradcam_images[i][dot["y_start"]:,dot_offset+dot["x_start"]:dot_offset+dot["x_end"],dot["channel"]] = intensity

            # result = Image.fromarray(gradcam_images[i].astype(np.uint8), mode="RGB")
            tmp = cv2.cvtColor(gradcam_images[i], cv2.COLOR_BGR2RGB)
            result = Image.fromarray(tmp.astype(np.uint8))
            result.save(root_dir+"/case"+case+"_"+str(i)+".png")

    f = open(root_dir+"/MASKVALScase"+case+".txt","w+")
    f.write(str(mask))
    f.close()


def find_temp_mask_red_dots(image_width, image_height, mask, round_up_mask):
    mask_len = len(mask)
    dot_width = int(image_width//(mask_len+4))
    dot_padding = int((image_width - (dot_width*mask_len))//mask_len)
    dot_height = int(image_height//20)
    dots = []
    
    for i,m in enumerate(mask):
        
        if(round_up_mask):
            if(mask[i]>0.5):
                mask[i]=1
            else:
                mask[i]=0
                
        dot={'y_start': -dot_height,
             'y_end' : image_height,
             'x_start' : i*(dot_width+dot_padding),
             'x_end' : i*(dot_width+dot_padding)+dot_width}
        if(mask[i]==0):
            dot['channel']=1  # Green
        else:
            dot['channel']=2  # in BGR.
            
        dots.append(dot)
        
    return dots
    

def visualize_results(orig_seq, pert_seq, mask, root_dir=None,
                      case="0", mark_imgs=True, iter_test=False):
    if(root_dir==None):
        root_dir= '/workspace/projects/spatiotemporal-interpretability/tensorflow/' + \
                  FLAGS.output_folder + "/"
    root_dir+="/PerturbImgs/"
        
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for i in range(orig_seq.shape[1]):

        if(mark_imgs):
            orig_seq[:, i, :10, :10, 1:]=0
            orig_seq[:, i, :10, :10, 0]=mask[i]*255
            pert_seq[:, i, :10, :10, 1:]=0
            pert_seq[:, i, :10, :10, 0]=mask[i]*255
        result = Image.fromarray(pert_seq[0,i,:,:,:].astype(np.uint8))
        result.save(root_dir+"case"+case+"pert"+str(i)+".png")
    f = open(root_dir+"case"+case+".txt","w+")
    f.write(str(mask))
    f.close()


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

    video_ID = parsed_features['video_id']

    images.set_shape([FLAGS.seq_length, FLAGS.image_height, FLAGS.image_width, 3])
    label.set_shape([NUM_CLASSES])

    return images, label, video_ID

def create_dataset(filepath):
    import os.path
    dataset = tf.data.TFRecordDataset(filepath)
    # print(filepath)
    # print(os.path.isfile(filepath))
    if FLAGS.shuffle_data == 'yes':
        print('Shuffling the training data...')
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
    dataset = dataset.map(map_func=parse_fn,
                          num_parallel_calls=FLAGS.nb_parallel_calls)
    dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
                                   padded_shapes=([FLAGS.seq_length,
                                                  FLAGS.image_height,
                                                  FLAGS.image_width, 3],
                                                  [NUM_CLASSES], ()))
    dataset = dataset.prefetch(FLAGS.batch_size)

    return dataset


def get_top_k(array, k):
    top_k = array.argsort()[-k:][::-1]
    return top_k


def create_image_arrays(input_sequence, gradcams, time_mask,
                        output_folder, video_ID, mask_type):

    combined_images = []
    for i in range(FLAGS.seq_length):
        input_data_img = input_sequence[0, i, :, :, :]

        time_mask_copy = time_mask.copy()

        combined_img = np.concatenate((np.uint8(input_data_img),
                                      np.uint8(gradcams[i]),
                                      np.uint8(perturb_sequence(
                                             input_sequence,
                                             time_mask_copy,
                                             perb_type=mask_type,
                                             snap_values=True)[0,i,:,:,:])),
                                      axis=1)

        combined_images.append(combined_img)
        cv2.imwrite(os.path.join(output_folder,
                                 "img%02d.jpg" % (i + 1)),
                                 combined_img)

    # combined_images = np.transpose(np.array(combined_images), (3,0,1,2))
    visualize_results_on_gradcam(combined_images,
                                 time_mask,
                                 root_dir=output_folder,
                                 case=mask_type+video_ID)
    
                    
    return combined_images

                    
def main(argv):

    df = pd.read_csv(FLAGS.clip_selection)  # DataFrame containing the clips to run on.

    # First we need to recreate the same variables as in the model.
    tf.reset_default_graph()
    seq_shape = (FLAGS.batch_size, FLAGS.seq_length, FLAGS.image_height, FLAGS.image_width, 3)
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

        def recurrence(last_value, current_elem):
            update_tensor = (1-mask_clip[current_elem])*original_input_var[:,current_elem,:,:,:] + \
                            mask_clip[current_elem]*last_value
            return update_tensor
        
        perturb_op = tf.scan(fn=recurrence,
                             elems=frame_inds,
                             initializer=original_input_var[:,0,:,:,:])
        perturb_op = tf.reshape(perturb_op, seq_shape)

    y = tf.placeholder(tf.float32, [FLAGS.batch_size, NUM_CLASSES])
    logits, clstm_3 = clstm.clstm(perturb_op, bn=False, is_training=False, num_classes=NUM_CLASSES)
    after_softmax = tf.nn.softmax(logits)
    
    
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
        saver.restore(sess, "/workspace/checkpoints/" + FLAGS.checkpoint_name)

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
        
        for ind, row in df.iterrows():

            # Get values
            subject = row['Subject']
            video_ID = row['Video_ID']
            label = row['Label']
            # Retrieve the frame paths for this clip
            paths = get_paths_for_sequence(subject, video_ID)
            # Prepare one "batch" of data (bs=1)
            input_var, label = data_for_one_sequence_5D(paths, label)

            # only look at cases where a certain class (if class of interest 'classoI' was given)
            current_class = np.argmax(label)
            print(current_class)


            if current_class in df['Label'].values.tolist() and video_ID in [derp for derp in df[df['Label']==current_class]['Video_ID'].values]:

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
                
                #Start mask optimization, should be
                #Loss = lam1*||Mask size|| + lam2*Beta loss + class_score
                
                if not have_output:
                    output = sess.run(after_softmax, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                     original_input_var: input_var,
                                     frame_inds: range(FLAGS.seq_length)})
                    have_output=True
                
                if(run_temp_mask):
                    if (maskType == 'gradient'):
                        start_mask = init_mask(input_var, mask_var, original_input_var,
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

                            perturbed_sequence = perturb_sequence(input_var, time_mask, perb_type='reverse')
                            
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
                                          input_var, label, target_index)

                    '''beginning of gradcam write to disk'''
                    

                    os.makedirs(save_path, exist_ok=True)
                    
                if(do_gradcam and run_temp_mask):
                    create_image_arrays(input_var, gradcam, time_mask,
                                        save_path, video_ID, 'freeze')

                    if FLAGS.temporal_mask_type == 'reverse':
                        # Also create the image arrays for the reverse operation.
                        create_image_arrays(input_var, gradcam, time_mask,
                                            save_path, video_ID, 'reverse')
                
                if(run_temp_mask):
                    visualize_results(input_var,
                                      perturb_sequence(input_var,
                                                      time_mask, perb_type='reverse'),
                                      time_mask,
                                      root_dir=save_path,
                                      case=video_ID, mark_imgs=True,
                                      iter_test=False)
    

if __name__ == '__main__':
    tf.app.run()
            
