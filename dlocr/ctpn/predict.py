import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dlocr.ctpn import default_ctpn_config_path, default_ctpn_weight_path
from dlocr.ctpn.core import post_process
from dlocr.ctpn.lib import utils
from dlocr.ctpn.model import get_model
from dlocr.utils import draw_rect


def predict(model, image, output_path=None, mode=1):
    if type(image) == str:
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img = image
    h, w, c = img.shape

    # image size length must be greater than or equals 16 x 16,
    # because of the image will be reduced by 16 times.
    if h < 16 or w < 16:
        transform_w = max(16, w)
        transform_h = max(16, h)
        transform_img = np.ones(shape=(transform_h, transform_w, 3), dtype='uint8') * 255
        transform_img[:h, :w, :] = img
        h = transform_h
        w = transform_w
        img = transform_img

    # zero-center by mean pixel
    m_img = img - utils.IMAGE_MEAN
    m_img = np.expand_dims(m_img, axis=0)

    cls, regr, cls_prod = model.predict_on_batch(m_img)

    text_rects = post_process(cls_prod, regr, h, w)

    if mode == 1:
        for i in text_rects:
            draw_rect(i, img)

        plt.imshow(img)
        plt.show()
        if output_path is not None:
            cv2.imwrite(output_path, img)
    elif mode == 2:
        return text_rects, img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="模型权重文件位置",
                        default=default_ctpn_weight_path)
    parser.add_argument("--output_file_path", help="标记文件保存位置",
                        default=None)

    args = parser.parse_args()

    image_path = args.image_path  # 图像位置
    weight_path = args.weights_file_path  # 模型权重位置
    output_file_path = args.output_file_path  # 保存标记文件位置

    m, pm = get_model()
    pm.load_weights(r"../weights/weights-ctpnlstm-init.hdf5")

    start_time = time.time()
    predict(pm, image_path, output_path=output_file_path)
    print("cost ", (time.time() - start_time) * 1000, " ms")
