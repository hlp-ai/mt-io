import argparse
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ocr.densenet.create_tfrecords import _bytes_feature

from ocr.ctpn.data_loader import DataLoader


def _serialize_example(img, t):
    img = img.astype(np.float32)
    serialized_img = tf.io.serialize_tensor(img)
    rpn_class = t["rpn_class"]
    rpn_class = rpn_class.astype(np.float32)
    serialized_class = tf.io.serialize_tensor(rpn_class)
    rpn_regress = t["rpn_regress"]
    rpn_regress = rpn_regress.astype(np.float32)
    serialized_regress = tf.io.serialize_tensor(rpn_regress)
    feature = {
        'img': _bytes_feature(serialized_img.numpy()),
        'rpn_class': _bytes_feature(serialized_class.numpy()),
        'rpn_regress': _bytes_feature(serialized_regress.numpy()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", required=True, help="data directory")
    arg_parser.add_argument("--tfrecord_file", required=True, help="output tfrecords file")
    args = arg_parser.parse_args()

    data_loader = DataLoader(os.path.join(args.data_dir, "Annotations"),
                             os.path.join(args.data_dir, "JPEGImages"))
    total = data_loader.total_size

    tfds_file = args.tfrecord_file

    with tf.io.TFRecordWriter(tfds_file) as tfds_f:
        for _ in tqdm(range(total)):
            img, t = next(data_loader.load_data())
            # print(img)
            # print(img.shape, t["rpn_class"].shape, t["rpn_regress"].shape)
            # print(img.dtype, t["rpn_class"].dtype, t["rpn_regress"].dtype)
            tf_example = _serialize_example(img, t)
            tfds_f.write(tf_example.SerializeToString())
