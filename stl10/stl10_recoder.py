import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import cv2
import numpy as np
import tensorflow as tf
import time
import typing
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def to_tfrecord(img, label, format: str = ".webp"):
    # buffer = img.tobytes()
    buffer = cv2.imencode(format, img)[1].tobytes()
    feature_dict = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH])),
        'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[CHANNEL])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def save_tfrecrd(images: typing.List[np.ndarray],
                 labels: typing.List[str],
                 filename: str,
                 format: str):
    tf_writer = tf.io.TFRecordWriter(filename)
    for i in range(len(images)):
        tf_example = to_tfrecord(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), labels[i], format)
        tf_writer.write(tf_example.SerializeToString())
    tf_writer.close()


def write_tfrecord(format, train_x, train_y, test_x, test_y):
    save_tfrecrd(train_x, train_y, f"stl10_train{format}.tfrecord", format)
    save_tfrecrd(test_x, test_y, f"stl10_test{format}.tfrecord", format)


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


def read_tfrecord(format):
    a = tf.data.TFRecordDataset(f'stl10_train{format}.tfrecord').map(decode_fn)
    datas = []
    for e in a:
        height = e['height']
        width = e['width']
        channel = e['channel']
        image = np.frombuffer(e['image'].numpy(), np.uint8)  # .reshape(height, width, channel)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        label = e['label']
        datas.append((image, label))
    return datas


def test_write():
    train_x = read_all_images("train_X.bin")
    train_y = read_labels("train_y.bin")
    test_x = read_all_images("test_X.bin")
    test_y = read_labels("test_y.bin")

    formats = [".bmp", ".jpg", ".png", ".webp", ".tiff"]
    colors = ["dodgerblue", "royalblue", "mediumblue"]
    bar_width = 0.8
    x = np.arange(len(formats))
    plots = []
    data_list = []

    for i in range(len(formats)):
        filename = f"stl10_{formats[i]}.tfrecord"
        start = time.time()
        write_tfrecord(formats[i], train_x, train_y, test_x, test_y)
        print(f"{formats[i]} time: {time.time() - start} Sec")
        filesize = (os.path.getsize(f"stl10_train{formats[i]}.tfrecord") + os.path.getsize(
            f"stl10_test{formats[i]}.tfrecord")) // 1024
        data_list.append(filesize)

    plt.bar(x=x, height=data_list, width=bar_width, color="dodgerblue")

    plt.ylabel('File Size(KB)', fontsize=16)
    plt.xlabel('Image File Formats', fontsize=16)
    plt.xticks(x, formats, fontsize=14)

    plt.legend(list(map(lambda x: x[0], plots)), ('Train', 'Valid', 'Test'), fontsize=15)
    plt.savefig("cmp_img_fmts.png", bbox_inches='tight')
    plt.show()
    # read_tfrecord()


def test_read():
    formats = [".bmp", ".jpg", ".png", ".webp", ".tiff"]
    for i in range(len(formats)):
        start = time.time()
        read_tfrecord(formats[i])
        print(f"{formats[i]} time: {time.time() - start} Sec")


if __name__ == "__main__":
    test_read()