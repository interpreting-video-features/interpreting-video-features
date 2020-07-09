from keras.preprocessing.image import load_img, img_to_array


def process_image(image_path):
    img = load_img(image_path)
    return img_to_array(img).astype(np.float32)


