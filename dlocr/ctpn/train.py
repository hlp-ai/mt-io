import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from dlocr.ctpn.data_loader import DataLoader
from dlocr.ctpn.model import get_model


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


train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = Adam(1e-05)


# @tf.function(input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32),
#                               {"rpn_class": tf.TensorSpec([1, None, None, 2], tf.float32),
#                                "rpn_regress": tf.TensorSpec([1, None, None, 2], tf.float32)}])
def train_step(images, gt):
    with tf.GradientTape() as tape:
        cls, regr = model(images)
        cls_loss = _rpn_loss_cls(gt["rpn_class"], cls)
        regr_loss = _rpn_loss_regr(gt["rpn_regress"], regr)
        loss = cls_loss + regr_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


if __name__ == "__main__":
    model, _ = get_model(vgg_weights_path="../weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    data_loader = DataLoader(r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations",
                             r"D:\dataset\ocr\VOCdevkit\VOC2007\JPEGImages")

    step = 1
    steps = 20
    for s, t in data_loader.load_data():
        train_step(s, t)

        print(step, train_loss.result().numpy())

        step += 1

        if step > steps:
            break
