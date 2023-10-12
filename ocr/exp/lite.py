import tensorflow as tf
import numpy as np


saved_path = "./ctpn/1"
saved_fn = "ocr_ctpn.tflite"

model = tf.saved_model.load(saved_path)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, None, None, 3])
# converter = TFLiteConverter.from_concrete_functions([concrete_func])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
tflite_model = converter.convert()

# Save the model.
with open(saved_fn, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=saved_fn)

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(len(input_details))
print(input_details[0])
# output_details = interpreter.get_output_details()
# print(output_details)
# print(len(output_details))
# signature_lists = interpreter.get_signature_list()
# print(signature_lists)

# Test the model on random input data.
input_shape = [1, 100, 200, 3]
# interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
interpreter.allocate_tensors()
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()