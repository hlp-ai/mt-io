import tensorflow as tf
from ocr.ctpn import get_model

predict_model = get_model()  # 创建模型, 不需加载基础模型权重
predict_model.load_weights(r"../weights/weights-ctpnlstm-init.hdf5")  # 加载模型权重

saved_path = "./ctpn/1"
tf.saved_model.save(predict_model, saved_path)

imported_ctpn = tf.saved_model.load(saved_path)
print(imported_ctpn.signatures)

input = tf.random.normal(shape=(1, 100, 200, 3))
print(imported_ctpn(input))
