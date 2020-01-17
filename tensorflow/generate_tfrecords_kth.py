import os
import sys
import cv2
import math
import random
import os.path
import argparse

from datetime import datetime

import numpy as np
import pandas as pd
import scipy.misc as sm
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _convert_to_sequential_example(video_id, video_buffer, label, args, nb_frames):
    """Build a SequenceExample proto for an example.
    Departed from https://stackoverflow.com/questions/48101576/tensorflow-read-video-frames-from-tfrecords-file
    Args:
        video_id: string, id for video file, e.g., '33467'
        video_buffer: numpy array with the video frames, with dims [n_frames, height, width, n_channels]
        label: integer, identifier for the ground truth class for the network
        args: args from argparse
    Returns:
        Example proto
    """
    assert len(video_buffer) == nb_frames
    assert video_buffer.shape[1] == args.height
    assert video_buffer.shape[2] == args.width

    features = {}
    features['nb_frames']   = _int64_feature(video_buffer.shape[0])
    features['height']      = _int64_feature(video_buffer.shape[1])
    features['width']       = _int64_feature(video_buffer.shape[2])
    features['label']       = _int64_feature(label)
    features['video_id']    = _bytes_feature(str.encode(video_id))
  
    # Compress the frames using JPG and store in as a list of strings in 'frames'
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                      for frame in video_buffer]
    features['frames'] = _bytes_list_feature(encoded_frames)
  
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def get_video_buffer(frame_folder_path, start_frame, end_frame):
    """Build a np.array of all frames in a sequence.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    
    images = []
    for f in range(start_frame, end_frame+1):
        counter_format = ("%02d" % (f))  # 01 etc.
        frame_path = frame_folder_path + 'frame' + counter_format + '.jpg'  # folder/frame01.jpg
        im = process_image(frame_path)
        images.append(im)
    frames = np.asarray(images)
    return frames


def get_list_of_sampled_frames(start_frame, end_frame, nb_frames_to_sample):
    frames = range(start_frame, end_frame+1)
    length = float(len(frames))
    sampled_frames = []
    for i in range(nb_frames_to_sample):
        sampled_frames.append(frames[int(math.ceil(i * length / nb_frames_to_sample))])
    return sampled_frames


def get_list_of_cohesive_frames(start_frame, end_frame, nb_frames_to_sample):
    frames = list(range(start_frame, end_frame+1))
    return frames


def get_fixed_number_of_frames_video_buffer(frame_folder_path, start_frame, end_frame, nb_frames):
    """Build a np.array of sampled frames from a sequence.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    print('Sampling {} frames from every clip...'.format(nb_frames)) 
    print('\n')
    images = []
    total_frames= (end_frame-start_frame)
    # If the clip has fewer frames than we want to sample
    if total_frames < nb_frames:
        assert total_frames > 0
        # Just return all frames
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    total_frames)
        # Repeat the last frame for the remaining number of frames
        last_frame =  sampled_frames[-1]
        diff = nb_frames - total_frames
        for i in range(diff):
            sampled_frames.append(last_frame)
    else:
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    nb_frames)

    for f in sampled_frames:
        counter_format = ("%02d" % (f))  # 01 etc.
        frame_path = frame_folder_path + 'frame' + counter_format + '.jpg'  # folder/frame01.jpg
        print(frame_path)
        im = process_image(frame_path)
        images.append(im)

    assert len(images) == nb_frames
    frames = np.asarray(images)

    return frames


def get_cohesive_crop_of_frames_video_buffer(frame_folder_path, start_frame, end_frame, nb_frames):
    """Build a np.array of a sampled, cohesive crop of x frames from a sequence of frames.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        start_frame: int, index of first frame.
        end_frame: int, index of last frame.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    print('Sampling {} cohesive frames from every clip...'.format(nb_frames)) 
    print('Start frame index: {}, end frame index: {}'.format(start_frame,
                                                              end_frame)) 
    print('\n')
    images = []
    total_frames= (end_frame-start_frame)
    # If the clip has fewer frames than we want to sample
    if total_frames < nb_frames:
        assert total_frames > 0
        # Just return all frames
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    total_frames)
        # Repeat the last frame for the remaining number of frames
        last_frame =  sampled_frames[-1]
        diff = nb_frames - total_frames
        for i in range(diff):
            sampled_frames.append(last_frame)
    else:
        sampled_frames = get_list_of_cohesive_frames(start_frame,
                                                     end_frame,
                                                     nb_frames)

    for f in sampled_frames:
        counter_format = ("%02d" % (f))  # 01 etc. This is how the frames were saved earlier.
        frame_path = frame_folder_path + 'frame' + counter_format + '.jpg'  # folder/frame01.jpg
        print(frame_path)
        im = process_image(frame_path)
        images.append(im)

    assert len(images) == nb_frames
    frames = np.asarray(images)

    return frames


def process_image(image_path):
    img = load_img(image_path)
    return img_to_array(img).astype(np.float32)


def process_files_and_write(df, args, mode='all', nb_frames_to_sample=None):
    subject = df.iloc[0]['subject']
    output_filename = 'kth_subject_' + str(subject) + '.tfrecords'
    output_file = os.path.join(args.output_folder, output_filename)
    print('Output file path: ', output_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    
    # Path to root folder for jpg frames
    frames_dir = '/data/kth_dataset/frames_per_subject/'

    for ind, row in df.iterrows():  # One row is for one clip.
        print('Index in df: ', ind, end='\r')
        video_id = str(row['clip_name'])
        label = row['label']
    
        # Paths to folder for jpg frames
        subject_folder_path = frames_dir + str(subject) + '/' 
        frames_folder_path = subject_folder_path + video_id + '/'

        for rep in range(1, 5):  # There are 4 repetitions of each action per clip.
            start_col = str(rep) + '_start'  # Dataframe-columns are called 1_start, 1_end, 2_start, 2_end, etc.
            end_col = str(rep) + '_end'
            if math.isnan(row[start_col]):
                continue
            if math.isnan(row[end_col]):
                continue
            start_frame = int(row[start_col])
            end_frame = int(row[end_col])
            total_frames = end_frame + 1 - start_frame
            if mode == 'all':
                nb_frames = total_frames
                video_buffer = get_video_buffer(frames_folder_path,
                                                start_frame,
                                                end_frame)
            if mode == 'sample':
                nb_frames = nb_frames_to_sample
                video_buffer = get_fixed_number_of_frames_video_buffer(frames_folder_path,
                                                                       start_frame,
                                                                       end_frame,
                                                                       nb_frames)
            if mode == 'sample_cohesive_crop':
                nb_frames = nb_frames_to_sample
                all_frame_inds = np.array(range(start_frame, end_frame+1))
                print(start_frame, end_frame)
                if total_frames > nb_frames:
                       start_frame = np.random.choice(all_frame_inds[:-nb_frames])
                       end_frame = start_frame + nb_frames - 1
                # Else just use the given start and end frames.
                video_buffer = get_cohesive_crop_of_frames_video_buffer(frames_folder_path,
                                                                        start_frame,
                                                                        end_frame,
                                                                        nb_frames)
            # Save as tfrecords
            example = _convert_to_sequential_example(video_id, video_buffer, label, args, nb_frames)
            writer.write(example.SerializeToString())

    writer.close()
        

def main():
    
    main_df = pd.read_csv('/data/kth_dataset/frames_labels_subjects_fixed.csv')
    
    parser = argparse.ArgumentParser(
        description='Some parameters.')
    parser.add_argument(
        '--output-folder', nargs='?', type=str,
        help='Folder where to output the tfrecords file')
    parser.add_argument(
        '--sample-mode', nargs='?', type=str,
        help='all|sample|sample_cohesive_crop')
    parser.add_argument(
        '--width', nargs='?', type=int,
        help='Image width')
    parser.add_argument(
        '--height', nargs='?', type=int,
        help='Image height')
    parser.add_argument(
        '--nb-to-sample', nargs='?', type=int,
        help='Fixed number of frames to sample per clip')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    print('Saving results to %s' % args.output_folder)

    # Run it and create one .tfrecords-file per subject!
    for s in range(1,26):  # 25 subjects
        subject_df = main_df[main_df['subject'] == s]
        process_files_and_write(subject_df,
                                args,
                                mode=args.sample_mode,
                                nb_frames_to_sample=args.nb_to_sample)


if __name__ == '__main__':
    main()
    print('\n')

