#import av
import torch
import numpy as np

from data_parser import PicDatabase
from data_augmentor import Augmentor
import torchvision
from transforms_video import *
from utils import save_images_for_debug
from PIL import Image

class ImLoader(torch.utils.data.Dataset):

    def __init__(self, root, json_file_input="", json_file_labels="", clip_size=16,
                 is_val=False, get_item_id=False, is_test=False):
        self.dataset_object = PicDatabase(root)#="/smth_smth/data_single/train/"
        self.path_data = self.dataset_object.input_data
        self.classes = self.dataset_object.classes

        self.root = root


        self.clip_size = clip_size
        self.nclips = 1
        self.step_size = 1
        self.is_val = is_val
        self.get_item_id = get_item_id

    def __getitem__(self, index):

        item = self.path_data[index]
        imgs = []
        for i in range(self.clip_size):
            im = Image.open(item[2]+"/frame"+"{:02d}".format(i+1)+".jpg")
            im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 3))    
            imgs.append(im_arr)
        data = torch.from_numpy(np.array(imgs)).float()

        # format data to torch
        #we have T,H,W,C
        #pytorch expects Batch,Channel, T, H, W
        data = data.permute(3, 0, 1, 2)
        if self.get_item_id:
            return (data, int(item.label), item.id)
        else:
            return (data, int(item.label))

    def __len__(self):
        return len(self.path_data)
