import argparse
import time
from concurrent.futures.thread import ThreadPoolExecutor

from PIL import Image
import numpy as np

from ocr.densenet.data_reader import load_dict_sp, single_img_process, process_imgs
from model_with_process import get_model_with_process


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


def predict(model, image, id_to_char, num_classes):
    if type(image) == str:
        img = Image.open(image)
    else:
        img = image

    # X = single_img_process(img)
    X = np.array([img])

    y_pred = model.predict(X)

    y_pred = y_pred.argmax(axis=2)
    out = decode_single_line(y_pred[0], num_classes, id_to_char)

    return out


def predict_multi(model, images, id_to_char, num_classes):
    X = process_imgs(images)
    y_pred = model.predict_on_batch(X)

    return decode(y_pred, num_classes, id_to_char)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置", required=True)
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--weights_file_path", help="模型权重文件位置", required=True)

    args = parser.parse_args()

    image_path = args.image_path
    dict_file_path = args.dict_file_path
    weight_path = args.weights_file_path

    id_to_char = load_dict_sp(dict_file_path, "UTF-8")
    print(id_to_char)

    densenet, _ = get_model_with_process(num_classes=len(id_to_char))
    densenet.load_weights(weight_path)

    densenet.summary()

    start = time.time()
    print(predict(densenet, image_path, id_to_char, len(id_to_char)))
    print("cost ", (time.time() - start) * 1000, " ms")

