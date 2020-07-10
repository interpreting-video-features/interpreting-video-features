import os
import math
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import generate_tfrecords as gtfr


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
                video_buffer = gtfr.get_video_buffer(frames_folder_path,
                                                start_frame,
                                                end_frame)
            if mode == 'sample':
                nb_frames = nb_frames_to_sample
                video_buffer = gtfr.get_fixed_number_of_frames_video_buffer(
                                                        frames_folder_path,
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
                video_buffer = gtfr.get_cohesive_crop_of_frames_video_buffer(
                                                         frames_folder_path,
                                                         start_frame,
                                                         end_frame,
                                                         nb_frames)
            # Save as tfrecords
            example = gtfr.convert_to_sequential_example(video_id,
                                                         video_buffer,
                                                         label,
                                                         args,
                                                         nb_frames)
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

