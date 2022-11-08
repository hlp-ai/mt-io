import argparse
import time

from dlocr.densenet import default_densenet_config_path
from dlocr.densenet import get_or_create
from dlocr.densenet.data_reader import load_dict_sp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置", required=True)
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_densenet_config_path)
    parser.add_argument("--weights_file_path", help="模型权重文件位置", required=True)

    args = parser.parse_args()

    image_path = args.image_path
    dict_file_path = args.dict_file_path
    weight_path = args.weights_file_path
    config_path = args.config_file_path

    id_to_char = load_dict_sp(dict_file_path, "UTF-8")
    print(id_to_char)

    densenet = get_or_create(config_path, weight_path)
    densenet.base_model.summary()

    start = time.time()
    print(densenet.predict(image_path, id_to_char))
    print("cost ", (time.time() - start) * 1000, " ms")
