import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Lambda, Bidirectional, GRU


def _reshape(x):
  b = tf.shape(x)
  x = tf.reshape(x, [b[0] * b[1], b[2], b[3]])  # (N x H, W, C)
  return x


def _reshape2(x):
  x1, x2 = x
  b = tf.shape(x2)
  x = tf.reshape(x1, [b[0], b[1], b[2], 256])  # (N, H, W, 256)
  return x


def _reshape3(x):
  b = tf.shape(x)
  x = tf.reshape(x, [b[0], b[1] * b[2] * 10, 2])  # (N, H x W x 10, 2)
  return x


class CTPN(tf.keras.Model):

  def __init__(self, vgg_weights_path=None, **kwargs):
    super().__init__(**kwargs)

    self.image_shape = (None, None, 3)  # 大小不定
    self.base_model = VGG16(weights=None, include_top=False, input_shape=self.image_shape)
    if vgg_weights_path is not None:  # 基础模型预训练微调
      self.base_model.load_weights(vgg_weights_path)
      self.base_model.trainable = True
    else:
      self.base_model.trainable = False

    self.rpn_conv1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='rpn_conv1')
    self.lambda_reshape = Lambda(_reshape, output_shape=(None, 512))
    self.blstm = Bidirectional(GRU(128, return_sequences=True, reset_after=False), name='blstm')
    self.lambda_reshape2 = Lambda(_reshape2, output_shape=(None, None, 256))
    self.lstm_fc = Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')

    self.rpn_class_origin = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class_origin')
    self.rpn_regress_origin = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress_origin')

    self.lambda_reshape31 = Lambda(_reshape3, output_shape=(None, 2), name='rpn_class')
    self.lambda_reshape32 = Lambda(_reshape3, output_shape=(None, 2), name='rpn_regress')

  def call(self, image, training=False, **kwargs):
    self.base_model(image)
    sub_output = self.base_model.get_layer('block5_conv3').output

    x = self.rpn_conv1(sub_output)
    tf.print(x.shape)

    x1 = self.lambda_reshape(x)  # output_shape not include batch dim, (N*H, W, C), C=512
    tf.print(x1.shape)
    # shape = tf.shape(x1)
    x1.set_shape((None, None, 512))

    x2 = self.blstm(x1)

    x3 = self.lambda_reshape2([x2, x])  # (N, H, W, C), C=256
    x3 = self.lstm_fc(x3)

    # 分类分支
    cls = self.rpn_class_origin(x3)
    # 高度回归分支
    regr = self.rpn_regress_origin(x3)

    cls = self.lambda_reshape31(cls)  # (N, H*W*10, 2)
    # cls_prod = Activation('softmax', name='rpn_cls_softmax')(cls)
    regr = self.lambda_reshape32(regr)  # (N, H*W*10, 2)

    return {"rpn_class": cls, "rpn_regress": regr}

  # @tf.function(experimental_relax_shapes=True,
  #       input_signature=[
  #           tf.TensorSpec([1, None, None, 3], dtype=tf.float32, name="image"),
  #       ])
  # def infer(self, image):
  #   return self(image)


ctpn = CTPN()  # 创建模型, 不需加载基础模型权重
ctpn(tf.random.normal(shape=[1, 200, 200, 3]))
# ctpn.load_weights(r"../weights/weights-ctpnlstm-init.hdf5")  # 加载模型权重
saved_path = "./ctpn/1"
# tf.saved_model.save(ctpn, saved_path)
ctpn.save(saved_path)

imported_ctpn = tf.saved_model.load(saved_path)
print(imported_ctpn.signatures)
