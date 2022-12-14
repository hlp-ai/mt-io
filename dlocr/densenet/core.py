from concurrent.futures import ThreadPoolExecutor

import tensorflow.keras.backend as K
import numpy as np
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, AveragePooling2D, ZeroPadding2D, Permute, \
    TimeDistributed, Flatten, Dense, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from dlocr.densenet.data_reader import single_img_process, process_imgs


def _ctc_loss(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_single_line(pred_text, nclass, id_to_char):
    char_list = []

    # pred_text = pred_text[np.where(pred_text != nclass - 1)[0]]
    print(pred_text)

    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (pred_text[i] != pred_text[i - 1]) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(id_to_char[pred_text[i]])
    return u''.join(char_list)


def decode(pred, nclass, id_to_char):
    lines = []

    pred_texts = pred.argmax(axis=2)

    with ThreadPoolExecutor() as executor:
        for line in executor.map(lambda pred_text: decode_single_line(pred_text, nclass, id_to_char), pred_texts):
            lines.append(line)

    return lines


def _dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = _conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb])  # concatenation of channels
        nb_filter += growth_rate
    return x, nb_filter


def _conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def _transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)

    return x, nb_filter


class DenseNetOCR:

    def __init__(self,
                 num_classes,
                 lr=0.0005,
                 image_height=32,
                 image_channels=1,
                 maxlen=50,
                 dropout_rate=0.2,
                 weight_decay=1e-4,
                 filters=64,
                 weight_path=None,
                 num_gpu=1):
        self.image_shape = (image_height, None, image_channels)
        self.lr = lr
        self.image_height, self.image_channels = image_height, image_channels
        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.filters = filters
        self.num_classes = num_classes
        self.num_gpu = num_gpu
        self.base_model, self.train_model, self.parallel_model = self.__build_model()
        if weight_path is not None:
            print("Loading model from", weight_path)
            self.base_model.load_weights(weight_path)

    def __build_model(self):
        input = Input(shape=self.image_shape, name="the_input")  # ???h, w, c???
        nb_filter = self.filters

        x = Conv2D(nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
                   use_bias=False, kernel_regularizer=l2(self.weight_decay))(input)  # (h/2, w/2, nb_filter)

        # output channels: 64 +  8 * 8 = 128
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)
        # output channels: 128, (h/4, w/4, 128)
        x, nb_filter = _transition_block(x, 128, self.dropout_rate, 2, self.weight_decay)

        # output channels: 128 + 8 * 8 = 192
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)
        # output channels: 192->128, (h/8, w/8, 128)
        x, nb_filter = _transition_block(x, 128, self.dropout_rate, 2, self.weight_decay)

        # output channels: 128 + 8 * 8 = 192
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)

        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

        x = Permute((2, 1, 3), name='permute')(x)  # (w/8, h/8, c)
        x = TimeDistributed(Flatten(), name='flatten')(x)  # (w/8, d)
        y_pred = Dense(self.num_classes, name='out', activation='softmax')(x)

        base_model = Model(inputs=input, outputs=y_pred)

        labels = Input(shape=(self.maxlen,), dtype='float32', name="the_labels")
        input_length = Input(shape=(1,), name="input_length", dtype='int64')
        label_length = Input(shape=(1,), name="label_length", dtype='int64')

        loss_out = Lambda(_ctc_loss, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

        model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

        parallel_model = model

        adam = Adam(self.lr)
        # output value used as loss
        parallel_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

        return base_model, model, parallel_model

    def predict(self, image, id_to_char):
        if type(image) == str:
            img = Image.open(image)
        else:
            img = image

        X = single_img_process(img)
        X = np.array([X])

        y_pred = self.base_model.predict(X)

        y_pred = y_pred.argmax(axis=2)
        out = decode_single_line(y_pred[0], self.num_classes, id_to_char)

        return out

    def predict_multi(self, images, id_to_char):
        X = process_imgs(images)
        y_pred = self.base_model.predict_on_batch(X)

        return decode(y_pred, self.num_classes, id_to_char)
