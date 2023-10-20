import tensorflow as tf
from keras.layers import Reshape
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Lambda, Bidirectional, GRU

from ocr.ctpn.data_loader import DataLoader


def get_model(image_channels=3, vgg_weights_path=None):
    image_shape = (None, None, image_channels)  # 大小不定
    base_model = VGG16(weights=None, include_top=False, input_shape=image_shape)
    if vgg_weights_path is not None:  # 基础模型预训练微调
        base_model.load_weights(vgg_weights_path)
        base_model.trainable = True
    else:
        base_model.trainable = False

    # 基础模型输入和输出
    input = base_model.input
    sub_output = base_model.get_layer('block5_conv3').output

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='rpn_conv1')(sub_output)
    # x1 = Lambda(_reshape, output_shape=(None, 512))(x)  # output_shape not include batch dim, (N*H, W, C), C=512
    b = tf.shape(x)
    x1 = tf.reshape(x, [b[0] * b[1], b[2], b[3]])  # (N x H, W, C)
    # x1.set_shape((None, None, 512))
    x1 = tf.ensure_shape(x1, (None, None, 512))

    x2 = Bidirectional(GRU(128, return_sequences=True, reset_after=False), name='blstm')(x1)
    b = tf.shape(x)
    x3 = tf.reshape(x2, [b[0], b[1], b[2], 256])  # (N, H, W, 256)
    x3 = Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')(x3)

    # 分类分支
    cls = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class_origin')(x3)
    # cls = Lambda(_reshape3, output_shape=(None, 2), name='rpn_class')(cls)  # (N, H*W*10, 2)
    b = tf.shape(cls)
    cls = tf.reshape(cls, [b[0], b[1] * b[2] * 10, 2], name='rpn_class')  # (N, H x W x 10, 2)
    # cls_prod = Activation('softmax', name='rpn_cls_softmax')(cls)

    # 高度回归分支
    regr = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress_origin')(x3)
    # regr = Lambda(_reshape3, output_shape=(None, 2), name='rpn_regress')(regr)  # (N, H*W*10, 2)
    b = tf.shape(regr)
    regr = tf.reshape(regr, [b[0], b[1] * b[2] * 10, 2], name='rpn_regress')  # (N, H x W x 10, 2)

    # predict_model = Model(input, [cls, regr, cls_prod])

    train_model = Model(input, [cls, regr])

    return train_model

    # return train_model, predict_model


if __name__ == "__main__":
    print(tf.config.functions_run_eagerly())
    m = get_model()
    m.summary()
    # print(m.output)
    # print(m.get_layer("rpn_regress"))

    data_loader = DataLoader(r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations",
                             r"D:\dataset\ocr\VOCdevkit\VOC2007\JPEGImages")

    img, t = next(data_loader.load_data())
    print(img.shape, t["rpn_class"].shape, t["rpn_regress"].shape)

    print()

    r = m(img)
    print(r)
