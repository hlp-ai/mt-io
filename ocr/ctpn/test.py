import tensorflow as tf
import numpy as np
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from ocr.ctpn.lib import utils3
from ocr.ctpn.lib.TextProposalConnectorOriented2 import TextProposalConnectorOriented
from ocr.ctpn.model import get_model


@tf.function
def post_process(cls_prod, regr, h, w):
    # 生成基础锚框
    anchor = utils3.gen_anchor([int(h / 16), int(w / 16)], 16)

    # 得到预测框
    bbox = utils3.bbox_transfor_inv(anchor, regr)

    # 切掉图片范围外的部分
    bbox = utils3.clip_box(bbox, [h, w])

    # score > 0.7
    fg = tf.where(cls_prod[0, :, 1] > utils3.IOU_SELECT)[:, 0]
    select_anchor = tf.gather(bbox, fg)
    select_score = tf.gather(cls_prod[0, :, 1], fg)
    select_anchor = tf.cast(select_anchor, tf.int32)

    # filter size
    keep_index = utils3.filter_bbox(select_anchor, 16)

    # nms
    select_anchor = tf.gather(select_anchor, keep_index)
    select_score = tf.gather(select_score, keep_index)
    select_score = tf.reshape(select_score, (-1, 1))
    nmsbox = tf.concat([tf.cast(select_anchor, tf.float32), select_score], axis=1)
    keep = utils3.nms(nmsbox, 1 - utils3.IOU_SELECT)
    select_anchor = tf.gather(select_anchor, keep)
    select_score = tf.gather(select_score, keep)

    # text line
    textConn = TextProposalConnectorOriented()
    text_rects = textConn.get_text_lines(select_anchor, select_score, [h, w])

    return text_rects

def predict(model, image, output_path=None, mode=1):
    if type(image) == str:
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img = image
    h, w, c = img.shape

    # image size length must be greater than or equals 16 x 16,
    # because of the image will be reduced by 16 times.
    if h < 16 or w < 16:  # 对过小图片进行补充
        transform_w = max(16, w)
        transform_h = max(16, h)
        transform_img = np.ones(shape=(transform_h, transform_w, 3), dtype='uint8') * 255
        transform_img[:h, :w, :] = img
        h = transform_h
        w = transform_w
        img = transform_img

    # 和训练数据一样规范化
    m_img = img - utils3.IMAGE_MEAN
    m_img = np.expand_dims(m_img, axis=0)

    # cls, regr, cls_prod = model.predict_on_batch(m_img)
    cls, regr = model.predict_on_batch(m_img)
    cls_prod = softmax(cls, axis=-1)

    # 计算文本框列表
    text_rects = post_process(cls_prod, regr, h, w)

    return text_rects



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", required=True, help="图像位置")
    parser.add_argument("--image_path", default=r"D:\datasets\ocr\Iran-Vehicle-plate-dataset\test\JPEGImages\192.jpg",
                        help="图像位置")
    parser.add_argument("--weights_file_path", help="模型权重文件位置",
                        default=r"../weights/weights-ctpnlstm-init.hdf5")
    parser.add_argument("--output_file_path", help="标记文件保存位置", default=None)

    args = parser.parse_args()

    image_path = args.image_path  # 图像位置
    weight_path = args.weights_file_path  # 模型权重位置
    output_file_path = args.output_file_path  # 保存标记文件位置

    predict_model = get_model()  # 创建模型, 不需加载基础模型权重
    predict_model.load_weights(weight_path)  # 加载模型权重
    result = predict(predict_model, image_path, output_path=output_file_path)
    print(result)


