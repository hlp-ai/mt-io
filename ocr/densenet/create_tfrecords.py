"""Preprocess labeled data into TFRecord format"""
import argparse
import io
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    if isinstance(value, type(tf.constant(0.0))):
        value = value.numpy()
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(img_path, label):
    img = open(img_path.numpy(), 'rb').read()  # raw image data
    feature = {
        'img': _bytes_feature(img),
        "path": _bytes_feature(img_path),
        'label': _bytes_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(img_path, label):
    tf_string = tf.py_function(
        serialize_example,
        (img_path, label),
        tf.string)
    return tf.reshape(tf_string, ())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("meta_file")
    arg_parser.add_argument("tfrecord_file")
    arg_parser.add_argument("num_examples", type=int)
    args = arg_parser.parse_args()

    meta_file = args.meta_file  # 标注文件
    max_chars = 32
    num_examples = args.num_examples  # 最多转换多少样本

    lines = io.open(meta_file, encoding="utf-8").readlines()
    lines = [line.strip() for line in lines]

    max_label_len = max([len(line.split("\t")[1]) for line in lines])  # 样本中文本最大长度
    print("Max len of label:", max_label_len)

    # filter by length of label
    if max_chars is not None:
        lines = list(filter(lambda line: len(line.split("\t")[1]) <= max_chars, lines))

    print("Shuffling...")
    np.random.shuffle(lines)

    # subset
    if num_examples is not None:
        lines = lines[:num_examples]


    def split(line):
        pair = tf.strings.split(line, "\t", maxsplit=1)
        return pair[0], pair[1]  # 图像文件, 文本


    # make dataset
    ds = tf.data.Dataset.from_tensor_slices(lines)  # 行数据集
    ds = ds.map(split)  # (图像文件, 文本)元组数据集

    # to tf.Example
    serialized_ds = ds.map(tf_serialize_example, num_parallel_calls=8)

    # write into TFRecords file
    filename = args.tfrecord_file
    writer = tf.io.TFRecordWriter(filename)
    count = 0
    for e in serialized_ds:
        writer.write(e.numpy())
        count += 1
        if count % 100 == 0:
            print(count)
    print(count)
