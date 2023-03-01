import tensorflow as tf
import numpy as np
from dlocr.ctpn.model import get_model


def convert_tflite(prediction_model, quantization):
  converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  if quantization == 'float16':
    converter.target_spec.supported_types = [tf.float16]
  # elif quantization == 'int8' or quantization == 'full_int8':
  #   converter.representative_dataset = representative_data_gen
  if quantization == 'full_int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
  tf_lite_model = converter.convert()
  open(f'ocr_ctpn.tflite', 'wb').write(tf_lite_model)


predict_model = get_model()  # 创建模型, 不需加载基础模型权重
predict_model.load_weights(r"../weights/weights-ctpnlstm-init.hdf5")  # 加载模型权重

convert_tflite(predict_model, "float16")

# # tf.saved_model.save(predict_model, "./saved")
# predict_model.save("./saved")
#
# # Convert the model.
# # converter = tf.lite.TFLiteConverter.from_keras_model(predict_model)
# converter = tf.lite.TFLiteConverter.from_saved_model("./saved")
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="ocr_ctpn.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(len(input_details))
print(input_details[0])
output_details = interpreter.get_output_details()
print(output_details)
print(len(output_details))
signature_lists = interpreter.get_signature_list()
print(signature_lists)

# Test the model on random input data.
input_shape = [1, 100, 200, 3]
# interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
interpreter.allocate_tensors()
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
