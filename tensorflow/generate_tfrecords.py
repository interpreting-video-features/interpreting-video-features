import os
import sys
import cv2
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


def _convert_to_sequential_example(video_id, video_buffer, label, args):
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
    assert len(video_buffer) == args.nb_frames
    assert video_buffer.shape[1] == args.width
    assert video_buffer.shape[2] == args.height

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


def get_video_buffer(frame_folder_path, nb_frames):
    """Build a np.array of all frames in a sequence.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    
    images = []
    for f in range(nb_frames):
        counter_format = ("%02d" % (f+1))  # 01 etc.
        frame_path = frame_folder_path + 'frame' + counter_format + '.jpg'  # folder/frame01.jpg
        im = process_image(frame_path)
        images.append(im)
    frames = np.asarray(images)
    return frames


def process_image(image_path):
    img = load_img(image_path)
    return img_to_array(img).astype(np.float32)


def process_files_and_write(df, labels_df, split_folder, args):

    output_filename = split_folder[:-1] + '.tfrecords'
    output_file = os.path.join(args.output_folder, output_filename)
    print('Output file path: ', output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    for ind, row in df.iterrows():
        print('Index in df: ', ind, end='\r')
        video_id = str(row['id'])
        label = row['template'].replace('[','').replace(']','')
        label_number = labels_df[label]
        label_folder_path = split_folder + str(label_number) + '/' 
        frames_folder_path = label_folder_path + video_id + '/'
        video_buffer = get_video_buffer(frames_folder_path, args.nb_frames)
        example = _convert_to_sequential_example(video_id, video_buffer, label_number, args)
        writer.write(example.SerializeToString())

    writer.close()
        

def main():
    train_json = '~/Downloads/something-something-v2-train.json'
    val_json = '~/Downloads/something-something-v2-validation.json'
    
    test_json = '~/Downloads/something-something-v2-test.json'
    
    labels_json = '~/Downloads/something-something-v2-labels.json'
    
    labels_df = pd.read_json(labels_json, typ='series')

    # SET VAL OR TRAIN

    train_df = pd.read_json(train_json)
    split_folder = 'train_128/'

    # val_df = pd.read_json(val_json)
    # split_folder = 'validation_128/'

    parser = argparse.ArgumentParser(
        description='Some parameters.')
    parser.add_argument(
        '--output-folder', nargs='?', type=str,
        help='Folder where to output the tfrecords file')
    parser.add_argument(
        '--nb-frames', nargs='?', type=int,
        help='The number of frames per sequence.')
    parser.add_argument(
        '--width', nargs='?', type=int,
        help='Image width')
    parser.add_argument(
        '--height', nargs='?', type=int,
        help='Image height')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    print('Saving results to %s' % args.output_folder)

    # Run it!
    process_files_and_write(train_df, labels_df, split_folder, args)


if __name__ == '__main__':
    main()
    print('\n')

