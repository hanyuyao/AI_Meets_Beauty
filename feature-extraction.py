import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATHC_SIZE = 128

def get_data(file_directory, list_directory, resize=False):
    image_dir = file_directory

    def _parse_function(file_path):
        image_string = tf.io.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        img = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        if resize:
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    # read all directories of files in image_dir
    # save a list of file names for searching
    path = []
    for dirpath, dirnames, imgnames in os.walk( image_dir ):
        imgnames = np.array(imgnames)
        np.save(list_directory, imgnames)
        for x in imgnames:
            path.append( os.path.join(dirpath,x) )
    path = np.array(path)
        
    file_path = tf.constant( path )
    dataset = tf.data.Dataset.from_tensor_slices( file_path )
    dataset = dataset.map(_parse_function)
    generator = dataset.batch( BATHC_SIZE )
    return generator


def build_model():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    base_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

    base_model.trainable = False

    return base_model


def build_model_1():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    base_model = tf.keras.applications.DenseNet169(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

    base_model.trainable = False

    return base_model


def generate_features():
    # generate features of 10000 images in dataset
    input_img = get_data('./data_10000_processed', './features/list_10000.npy')
    model = build_model()
    features = model.predict(input_img, verbose=1)
    features = np.array(features)
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]*features.shape[3] ))
    # features.shape = (10000, 100352)
    model_1 = build_model_1()
    features_1 = model_1.predict(input_img, verbose=1)
    features_1 = np.array(features_1)
    features_1 = features_1.reshape((features_1.shape[0], features_1.shape[1]*features_1.shape[2]*features_1.shape[3] ))
    # features_1.shape = (10000, 81536)
    f = np.concatenate((features,features_1), axis=1)
    # f.shape = (10000, 181888)
    np.save('./features/features_10000.npy', f)


def pred_features():
    # get features of input images
    input_img = get_data('./data_test', './features/list_test.npy', True)
    model = build_model()
    features = model.predict(input_img, verbose=1)
    features = np.array(features)
    features = features.reshape((features.shape[0],  features.shape[1]*features.shape[2]*features.shape[3] ))
    model_1 = build_model_1()
    features_1 = model_1.predict(input_img, verbose=1)
    features_1 = np.array(features_1)
    features_1 = features_1.reshape((features_1.shape[0], features_1.shape[1]*features_1.shape[2]*features_1.shape[3] ))
    f = np.concatenate((features,features_1), axis=1)
    np.save('./features/features_test.npy', f)


if __name__ == '__main__':
    tf.keras.backend.clear_session()

    # generate_features()
    pred_features()
