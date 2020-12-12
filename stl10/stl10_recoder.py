import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import cv2
import numpy as np
import tensorflow as tf

HEIGHT = 96
WIDTH = 96
CHANNEL = 3
# https://github.com/mttk/STL10/blob/master/stl10_input.py
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

TRAIN_IMG_PATH = 'train_X.bin'
TRAIN_LBL_PATH = 'train_y.bin'


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, CHANNEL, HEIGHT, WIDTH))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_tfrecord(img, label):
    # buffer = img.tobytes()
    buffer = cv2.imencode('.PNG', img)[1].tobytes()
    feature_dict = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH])),
        'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[CHANNEL])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def save_tfrecrd(images, labels, filename):
    tf_writer = tf.io.TFRecordWriter(filename)
    for i in range(len(images)):
        tf_example = to_tfrecord(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), labels[i])
        tf_writer.write(tf_example.SerializeToString())
    tf_writer.close()


def write_tfrecord():
    train_X = read_all_images("train_X.bin")
    train_y = read_labels("train_y.bin")
    print(train_X.shape)
    print(train_y.shape)
    save_tfrecrd(train_X, train_y, "stl10_train.tfrecord")
    test_X = read_all_images("test_X.bin")
    test_y = read_labels("test_y.bin")
    print(test_X.shape)
    print(test_y.shape)
    save_tfrecrd(test_X, test_y, "stl10_test.tfrecord")


def decode_fn(record_bytes):
    feature_dict = {
        'height': tf.io.FixedLenFeature([], dtype=tf.int64),
        'width': tf.io.FixedLenFeature([], dtype=tf.int64),
        'channel': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    return tf.io.parse_single_example(
        record_bytes,
        feature_dict
    )


def read_tfrecord():
    a = tf.data.TFRecordDataset('stl10_train.tfrecord').map(decode_fn)
    for e in a:
        height = e['height']
        width = e['width']
        channel = e['channel']
        image = np.frombuffer(e['image'].numpy(), np.uint8)  # .reshape(height, width, channel)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        label = e['label']
        print(image.shape)
        print(label)
        image = cv2.resize(image, (256, 256))
        cv2.imshow("img", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    write_tfrecord()
    # read_tfrecord()
