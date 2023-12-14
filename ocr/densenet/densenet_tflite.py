import os
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential

from ocr.densenet import get_model


def preprocess_image_layer():
    def single_img_process_tf(img):
        img = tf.image.rgb_to_grayscale(img)

        original_shape = tf.shape(img)
        original_height = original_shape[1]
        original_width = original_shape[2]
        new_width = original_width * 32 // original_height

        img = tf.image.resize(img, [32, new_width])
        img = tf.cast(img, tf.float32)

        img = img / 255.0 - 0.5

        img = tf.expand_dims(img, -1)

        return img

    return Lambda(single_img_process_tf, name='preprocessing_layer')

# Load your existing model
predict_model, train_model = get_model()

# Create a new model that includes the preprocessing layer
model_with_preprocessing = Sequential([
    preprocess_image_layer(),
    predict_model
])

model_with_preprocessing.build((None, None, None, 3))
model_with_preprocessing.summary()

tf.saved_model.save(model_with_preprocessing, 'saved_model')

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TF ops
    ]
tflite_model = converter.convert()

with open('desenet.tflite', 'wb') as f:
    f.write(tflite_model)


model_path = os.path.join(os.getcwd(), 'desenet.tflite')
print('TF Lite model:', model_path)


input_image = tf.random.normal(shape=(1, 200, 500, 3), dtype=tf.float32)
# Run inference with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=model_path)

input_details = interpreter.get_input_details()
interpreter.resize_tensor_input(input_details[0]['index'], (1, 200, 500, 3), strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], input_image)

print(input_details)

interpreter.invoke()
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
print(output.shape)
print(output)
