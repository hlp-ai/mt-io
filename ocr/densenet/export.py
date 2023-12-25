from model_with_process import get_model_with_process
import tensorflow as tf
from ocr.densenet.data_reader import OCRDataset, load_dict_sp

# 获取模型加载权重
id_to_char = load_dict_sp("../dictionary/latin_chars.txt", "UTF-8")
predict_model, train_model = get_model_with_process(num_classes=len(id_to_char))
predict_model.summary()
predict_model.load_weights("./densenet.hdf5")

# 保存模型
export_path = "./saved_model"
tf.saved_model.save(predict_model, export_path)

# 将saved_model转换为tflite格式
converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TF ops
    ]
tflite_model = converter.convert()
with open('./model.tflite', 'wb') as f:
    f.write(tflite_model)

