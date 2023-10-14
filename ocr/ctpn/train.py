import os

import tensorflow.keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from ocr.ctpn.data_loader import DataLoader
from ocr.ctpn.model import get_model


def _rpn_loss_regr(y_true, y_pred):
    """
    smooth L1 loss

    y_ture [1][HXWX10][3] (class,regr)
    y_pred [1][HXWX10][2] (reger)
    """
    sigma = 9.0

    y_true = tf.cast(y_true, "float32")

    cls = y_true[0, :, 0]
    regr = y_true[0, :, 1:3]
    regr_keep = tf.where(K.equal(cls, 1))[:, 0]
    regr_true = tf.gather(regr, regr_keep)
    regr_pred = tf.gather(y_pred[0], regr_keep)
    diff = tf.abs(regr_true - regr_pred)
    less_one = tf.cast(tf.less(diff, 1.0 / sigma), 'float32')
    loss = less_one * 0.5 * diff ** 2 * sigma + tf.abs(1 - less_one) * (diff - 0.5 / sigma)
    loss = K.sum(loss, axis=1)

    return K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.0))


def _rpn_loss_cls(y_true, y_pred):
    """
    softmax loss

    y_true [1][1][HXWX10] class
    y_pred [1][HXWX10][2] class
    """
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="D:/dataset/ocr/VOCdevkit/VOC2007", help="训练数据位置")
    parser.add_argument("--weights_file_path", default="./ctpn_weights.hdf5", help="模型权重文件位置")
    parser.add_argument("--vgg16_weights_path",
                        default="../weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                        help="VGG16权重文件路径")

    args = parser.parse_args()

    save_path = args.weights_file_path
    model = get_model(vgg_weights_path=args.vgg16_weights_path)
    model.summary()
    if os.path.exists(save_path):
        print("Loading model for training...")
        model.load_weights(save_path)

    print("Loading training data...")
    data_path = args.data_path
    data_loader = DataLoader(os.path.join(data_path, "Annotations"), os.path.join(data_path, "JPEGImages"))

    model.compile(optimizer=Adam(1e-05),
                        loss={'rpn_regress': _rpn_loss_regr, 'rpn_class': _rpn_loss_cls},
                        loss_weights={'rpn_regress': 1.0, 'rpn_class': 1.0})

    checkpoint = ModelCheckpoint(save_path, model=model, save_weights_only=True, monitor='loss',
                                 save_freq=500, save_best_only=True)
    earlystop = EarlyStopping(patience=2, monitor='loss')

    model.fit(data_loader.load_data(), epochs=1, steps_per_epoch=data_loader.steps_per_epoch,
              callbacks=[checkpoint, earlystop])
