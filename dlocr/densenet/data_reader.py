import sys
from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO

import tensorflow as tf
import numpy as np
from PIL import Image


feature_description = {
    'img': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'path': tf.io.FixedLenFeature([], tf.string),
}


def _parse_function(example_proto):
    """Parse example (a dict) from serialized string"""
    return tf.io.parse_single_example(example_proto, feature_description)


def single_img_process(img):
    """Preprocess image"""
    im = img.convert('L')  # to gray scale, 0 black, 255 white

    scale = im.size[1] * 1.0 / 32  # scale width according to height of 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w, 32), Image.ANTIALIAS)  # height 32, width non-fixed

    img = np.array(im).astype(np.float32) / 255.0 - 0.5  # [-0.5, 0.5]
    img = img.reshape((32, w, 1))
    return img


def _pad_img(img, len, value):
    out = np.ones(shape=(32, len, 1)) * value
    out[:, :img.shape[1], :] = img
    return out


def process_imgs(imgs):
    tmp = []
    with ThreadPoolExecutor() as executor:
        for img in executor.map(single_img_process, imgs):
            tmp.append(img)

    max_len = max([img.shape[1] for img in tmp])

    output = []
    with ThreadPoolExecutor() as executor:
        for img in executor.map(lambda img: _pad_img(img, max_len, 0.5), tmp):  # pad value 0.5 represent white
            output.append(img)

    return np.array(output)


def load_dict_sp(dict_file_path, encoding="utf-8", blank_first=True):
    """加载带空格的字典，由ID到字符映射"""
    with open(dict_file_path, encoding=encoding, mode='r') as f:
        chars = list(map(lambda char: char.strip('\r\n'), f.readlines()))

    chars[-1] = " "  # 最后一个为空格字符

    if blank_first:
        chars = chars[1:] + ['blank']  # 'blank' is a meta label used for the boundary of characters and used by ctc model

    dic = {i: v for i, v in enumerate(chars)}

    return dic


class OCRDataset(object):

    def __init__(self, dict_file, record_file, max_label_len=32):
        """

        Args:
            dict_file: 词典文件
            record_file: TFRecord文件
            max_label_len: 文本最大长度
        """
        self.id2char = load_dict_sp(dict_file)
        self.char2id = {c: i for i, c in self.id2char.items()}

        self.record_file = record_file
        self.max_label_len = max_label_len

    def encode(self, img_bytes, label):
        """Get input for model"""
        image_label = label.numpy().decode()
        label_len = np.array([len(image_label)])

        label = np.ones([self.max_label_len], dtype=np.int) * 9999  # 9999 is an id that not exists in dictionary
        label[0: len(image_label)] = [self.char2id[c] for c in image_label]

        img = Image.open(BytesIO(img_bytes.numpy()))

        IMAGE_WIDTH = 280  # Fiexed with of image
        SUBSAMPLE_MULTIPLIER = 8  # Subsampling multiplier of model for CTC input length
        input_len = np.array([IMAGE_WIDTH // SUBSAMPLE_MULTIPLIER])

        return single_img_process(img), label, input_len, label_len

    def tf_encode(self, e):
        return tf.py_function(self.encode, (e["img"], e["label"]), (tf.float32, tf.int64, tf.int64, tf.int64))

    @staticmethod
    def tf_xy(img, label, input_len, label_len):
        """Get the real training input for model"""
        return {'the_input': img,
                'the_labels': label,
                'input_length': input_len,
                'label_length': label_len}, {'ctc': 0.0}  # ctc is just a loss placeholder, whose value not matter

    def get_ds(self, batch_size=2, prefetch_size=6400):
        """Get dataset"""
        filenames = [self.record_file]
        raw_dataset = tf.data.TFRecordDataset(filenames)

        parsed_dataset = raw_dataset.map(_parse_function)  # parse serialized string into example
        # print(parsed_dataset)

        parsed_dataset = parsed_dataset.map(self.tf_encode)
        # print(parsed_dataset)

        parsed_dataset = parsed_dataset.map(self.tf_xy)

        parsed_dataset = parsed_dataset.shuffle(buffer_size=prefetch_size).batch(batch_size,
                                                                                 drop_remainder=True).prefetch(
            prefetch_size)

        return parsed_dataset


if __name__ == "__main__":
    tfr_fn = sys.argv[1]
    ocr_data = OCRDataset("../dictionary/char_std_5991.txt", tfr_fn)
    ds = ocr_data.get_ds(prefetch_size=4)
    print(ds.element_spec)
    for e in ds.take(1):
        print(e[0]["the_input"].numpy().min(), e[0]["the_input"].numpy().max())
        print(e)
