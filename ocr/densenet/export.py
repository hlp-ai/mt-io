from ocr.densenet.model_with_process import get_model_with_process
import tensorflow as tf
from ocr.densenet.data_reader import load_dict_sp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--weights_file_path", default="./densenet.hdf5")
    parser.add_argument("--export_path", default="./saved_model")
    parser.add_argument("--tflite_path", default="./densenet.tflite")
    args = parser.parse_args()

    # 获取模型加载权重
    dict_file_path = args.dict_file_path
    weights_file_path = args.weights_file_path
    id_to_char = load_dict_sp(dict_file_path, "UTF-8")
    predict_model, train_model = get_model_with_process(num_classes=len(id_to_char))
    predict_model.summary()
    predict_model.load_weights(weights_file_path)

    # 保存模型
    export_path = args.export_path
    tf.saved_model.save(predict_model, export_path)

    # 将saved_model转换为tflite格式
    converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TF ops
    ]
    tflite_model = converter.convert()
    tflite_path = args.tflite_path
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
