import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, AveragePooling2D, ZeroPadding2D, \
    Permute, \
    TimeDistributed, Flatten, Dense, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class ProcessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ProcessLayer, self).__init__(**kwargs)

    @tf.function
    def build(self, input_shape):
        super(ProcessLayer, self).build(input_shape)

    @tf.function(
        experimental_relax_shapes=True,
    )
    def call(self, input_img, **kwargs):
        if tf.shape(input_img)[-1] == 3:
            print("convert to gray image")
            input_img = tf.image.rgb_to_grayscale(input_img)
        # input_img = tf.image.rgb_to_grayscale(input_img)

        original_shape = tf.shape(input_img)
        original_height = original_shape[1]
        original_width = original_shape[2]
        new_width = original_width * 32 // original_height

        input_img = tf.image.resize(input_img, [32, new_width])
        input_img = tf.cast(input_img, tf.float32)

        input_img = input_img / 255.0 - 0.5

        # input_img = tf.expand_dims(input_img, -1)
        # 将input_img的channel改为1
        input_img = tf.reshape(input_img, [original_shape[0], 32, new_width, 1])

        return input_img


class ModelWithPreprocessing(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(ModelWithPreprocessing, self).__init__(**kwargs)
        self.model = model
        self.process_layer = ProcessLayer()

    def call(self, input_img, **kwargs):
        input_img = self.process_layer(input_img)
        return self.model(input_img)


def _ctc_loss(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


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


def _transition_block(input, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x, nb_filter


def get_model_with_process(num_classes=5991,
                           filters=64,
                           image_height=None,
                           image_channels=None,
                           dropout_rate=0.2,
                           weight_decay=1e-4,
                           lr=0.0005,
                           maxlen=50,
                           ):
    image_shape = (image_height, None, image_channels)
    input = Input(shape=image_shape, name="the_input")  # （h, w, c）
    x = ProcessLayer()(input)
    nb_filter = filters

    x = Conv2D(nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)  # (h/2, w/2, nb_filter)

    # output channels: 64 +  8 * 8 = 128
    x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, weight_decay)
    # output channels: 128, (h/4, w/4, 128)
    x, nb_filter = _transition_block(x, 128, dropout_rate, weight_decay)

    # output channels: 128 + 8 * 8 = 192
    x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, weight_decay)
    # output channels: 192->128, (h/8, w/8, 128)
    x, nb_filter = _transition_block(x, 128, dropout_rate, weight_decay)

    # output channels: 128 + 8 * 8 = 192
    x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)  # (w/8, h/8, c)
    x = TimeDistributed(Flatten(), name='flatten')(x)  # (w/8, d)
    y_pred = Dense(num_classes, name='out', activation='softmax')(x)

    predict_model = Model(inputs=input, outputs=y_pred)

    labels = Input(shape=(maxlen,), dtype='float32', name="the_labels")
    input_length = Input(shape=(1,), name="input_length", dtype='int64')
    label_length = Input(shape=(1,), name="label_length", dtype='int64')

    loss_out = Lambda(_ctc_loss, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

    train_model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    adam = Adam(lr)
    # output value used as loss
    train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

    return predict_model, train_model
