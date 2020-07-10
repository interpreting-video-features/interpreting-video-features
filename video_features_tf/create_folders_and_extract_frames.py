import pandas as pd
import subprocess
import argparse
import os


def iterate_over_videos_in_df_and_make_folders(df, split_folder):
    for ind, row in df.iterrows():
        print('Index in df: ', ind, end='\r')
        video_id = row['id']
        # print(video_id)
        label = row['template'].replace('[','').replace(']','')
        # print(label)
        label_number = labels_df[label]
        # print(label_number, '\n')
        label_folder_path = split_folder + str(label_number) + '/' 
        frames_folder_path = label_folder_path + str(video_id) + '/'

        clip_path = str(video_id) + '.webm'
        scale_str = 'scale=' + str(args.width) + ':' + str(args.height)

        # Make the folder for this split, if it does not exist
        if os.path.isdir(split_folder) == True:
            pass
        else:
            subprocess.call(['mkdir', split_folder])

        # Make the folder for this label, if it does not exist
        if os.path.isdir(label_folder_path) == True:
            pass
        else:
            subprocess.call(['mkdir', label_folder_path])

        # Make the folder for the frames of this clip
        subprocess.call(['mkdir', frames_folder_path])

        full_output_path = frames_folder_path + args.output_path

        # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 33191.webm

        ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                           '-of', 'default=noprint_wrappers=1:nokey=1', clip_path]

        duration = subprocess.check_output(ffprobe_command)
        print(duration)
        frame_rate = float(args.nb_frames)/float(duration)
        print(frame_rate)

        ffmpeg_command = ['ffmpeg', '-i', clip_path, '-vf', scale_str, '-r', str(frame_rate),
                          '-frames:v', args.nb_frames, full_output_path, '-hide_banner']
        print(ffmpeg_command)
        subprocess.call(ffmpeg_command)

    
train_json = '~/Downloads/something-something-v2-train.json'
val_json = '~/Downloads/something-something-v2-validation.json'

test_json = '~/Downloads/something-something-v2-test.json'

labels_json = '~/Downloads/something-something-v2-labels.json'


train_df = pd.read_json(train_json)
# val_df = pd.read_json(val_json)
labels_df = pd.read_json(labels_json, typ='series')

print('Parsing args...')
parser = argparse.ArgumentParser(description='Specify clip to extract frames from.')
parser.add_argument('--output-path', nargs='?', type=str,
    help='Output path without folder, typically on format frame[pp-sign]04d.jpg.')
parser.add_argument('--nb-frames', nargs='?', type=str,
    help='Fixed number of frames to extract from clip at specified frame rate')
parser.add_argument('--width', nargs='?', type=str,
    help='Width of frames to extract')
parser.add_argument('--height', nargs='?', type=str,
    help='Height of frames to extract')

args = parser.parse_args()
print(args)

scale_str = 'scale=' + args.width + ':' + args.height
iterate_over_videos_in_df_and_make_folders(train_df, 'train_128/')
# iterate_over_videos_in_df_and_make_folders(val_df, 'validation_128/')
print('\n')

