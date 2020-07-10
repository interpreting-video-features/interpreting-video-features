from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def process_image(image_path):
    img = load_img(image_path)
    return img_to_array(img).astype(np.float32)


def get_top_k(array, k):
    top_k = array.argsort()[-k:][::-1]
    return top_k

