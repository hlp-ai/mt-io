import numpy as np
import xmltodict
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

anchor_scale = 16

IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr  can find from  here https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]

DEBUG = True


def readxml(path):
    gtboxes = []
    with open(path, 'rb') as f:
        xml = xmltodict.parse(f)
        bboxes = xml['annotation']['object']
        if (type(bboxes) != list):
            x1 = bboxes['bndbox']['xmin']
            y1 = bboxes['bndbox']['ymin']
            x2 = bboxes['bndbox']['xmax']
            y2 = bboxes['bndbox']['ymax']
            gtboxes.append((int(x1), int(y1), int(x2), int(y2)))
        else:
            with ThreadPoolExecutor() as executor:
                for x1, y1, x2, y2 in executor.map(lambda bbox: (bbox['bndbox']['xmin'], bbox['bndbox']['ymin'],
                                                                 bbox['bndbox']['xmax'], bbox['bndbox']['ymax']),
                                                   bboxes):
                    gtboxes.append((int(x1), int(y1), int(x2), int(y2)))

        imgfile = xml['annotation']['filename']
    return np.array(gtboxes), imgfile


def gen_anchor(featuresize, scale):
    """生成锚框

    Args:
        featuresize: 图像特征图大小，shape of (h, w)
        scale: 图像下采样倍数, 16

    Returns:
        shape of (h*w*10, 4)
    """

    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    heights = tf.reshape(tf.constant(heights, dtype=tf.float32), (-1, 1))
    widths = tf.reshape(tf.constant(widths, dtype=tf.float32), (-1, 1))

    # 锚框大小为16像素
    base_anchor = tf.constant([0, 0, 15, 15], dtype=tf.float32)  # (xmin, ymin, xmax, ymax)

    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5

    # 一组十个锚框
    base_anchor = tf.concat([x1, y1, x2, y2], axis=1)

    h, w = featuresize
    shift_x = tf.range(0, w) * scale  # 原始图中x坐标位置点，每隔16个像素一个位置点
    shift_y = tf.range(0, h) * scale

    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])

    return tf.reshape(tf.stack(anchor), (-1, 4))

def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])  # (Nsample, 1)
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])  # (Msample, 1)

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))  # (Nsample, Msample)

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
    anchors: (Nsample, 4)
    gtboxes: (Nsample, 4)
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5  # (Nsample, )
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5  # (Nsample, )
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0  # (Nsample, )
    ha = anchors[:, 3] - anchors[:, 1] + 1.0  # (Nsample, )

    Vc = (Cy - Cya) / ha  # (Nsample, )
    Vh = np.log(h / ha)  # (Nsample, )

    ret = np.vstack((Vc, Vh))

    return ret.transpose()  # (Nsample, 2)


def bbox_transfor_inv(anchor, regr):
    """
    anchor: (NSample, 4)
    regr: (NSample, 2)

    根据锚框和偏移量反向得到GTBox
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5  # 锚框y中心点
    ha = anchor[:, 3] - anchor[:, 1] + 1  # 锚框高度

    Vcx = regr[:, :, 0]  # y中心点偏移
    Vhx = regr[:, :, 1]  # 高度偏移
    Cyx = Vcx * ha + Cya  # GTBox y中心点
    hx = tf.exp(Vhx) * ha  # GTBox 高
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5  # 锚框x中心点
    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    y1 = tf.squeeze(y1)
    y2 = tf.squeeze(y2)
    bbox = tf.stack([x1, y1, x2, y2], axis=1)
    return bbox



def clip_box(bbox, im_shape):
    bbox = tf.cast(bbox, dtype=tf.float32)
    im_shape = tf.cast(im_shape, dtype=tf.float32)
    # x1 >= 0
    bbox_x1 = tf.maximum(tf.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox_y1 = tf.maximum(tf.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox_x2 = tf.maximum(tf.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox_y2 = tf.maximum(tf.minimum(bbox[:, 3], im_shape[0] - 1), 0)
    clipped_bbox = tf.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)
    return clipped_bbox



def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    mask = tf.logical_and(tf.greater_equal(ws, minsize), tf.greater_equal(hs, minsize))
    keep = tf.cast(tf.where(mask)[:, 0], dtype=tf.int32)
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    """
    gtboxes: (Msample, 4)
    """
    imgh, imgw = imgsize

    # 产生候选锚框
    base_anchor = gen_anchor(featuresize, scale)  # (Nsample, 4)

    # 计算每个锚框和每个真实框间的IOU
    overlaps = cal_overlaps(base_anchor, gtboxes)  # 锚框和真实框的IOU, (Nsample, Msample)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])  # (Nsample,)
    labels.fill(-1)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # 和每个真实框IOU最大的锚框, (Msample, )

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)  # 和每个锚框最大的真实框, (Nsample, )
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]  # (Nsample, )

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # 剔除掉多余的正负样例
    # subsample positive labels, if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > RPN_POSITIVE_NUM:
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # subsample negative labels
    bg_index = np.where(labels == 0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
    if len(bg_index) > num_bg:
        # print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # 计算锚框、和锚框最大重叠的真实框之间的垂直高度和位置偏差
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])  # (Nsample, 2)

    return [labels, bbox_targets], base_anchor


class random_uniform_num():
    """
    uniform random
    """

    def __init__(self, total, start=0):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = start

    def get(self, batch_size):
        ret = []
        if self.index + batch_size > self.total:
            piece1 = self.range[self.index:]
            np.random.shuffle(self.range)
            self.index = (self.index + batch_size) - self.total
            piece2 = self.range[0:self.index]
            ret.extend(piece1)
            ret.extend(piece2)
        else:
            ret = self.range[self.index:self.index + batch_size]
            self.index = self.index + batch_size
        return ret

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    x2_1 = tf.add(x2, 1.0)
    y2_1 = tf.add(y2, 1.0)
    areas = tf.multiply(tf.add(x2_1, tf.negative(x1)), tf.add(y2_1, tf.negative(y1)))
    _, order = tf.nn.top_k(scores, k=tf.shape(scores)[0], sorted=True)

    keep = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    cond = lambda i, order, keep: tf.greater(tf.size(order), 0)

    def body(i, order, keep):
        i = order[0]
        keep = keep.write(keep.size(), i)
        xx1 = tf.maximum(x1[i], tf.gather(x1, order[1:]))
        yy1 = tf.maximum(y1[i], tf.gather(y1, order[1:]))
        xx2 = tf.minimum(x2[i], tf.gather(x2, order[1:]))
        yy2 = tf.minimum(y2[i], tf.gather(y2, order[1:]))

        xx2 = tf.add(xx2, 1.0)
        yy2 = tf.add(yy2, 1.0)
        w = tf.maximum(0.0, tf.add(xx2, tf.negative(xx1)))
        h = tf.maximum(0.0, tf.add(yy2, tf.negative(yy1)))
        inter = tf.multiply(w, h)

        area1 = tf.add(areas[i], tf.gather(areas, order[1:]))
        ovr = tf.divide(inter, tf.add(area1, tf.negative(inter)))
        inds = tf.where(tf.less_equal(ovr, thresh))[:, 0]
        order = tf.gather(order, tf.add(inds, 1))
        return tf.add(i, 1), order, keep

    _, _, keep = tf.while_loop(cond, body, [0, order, keep])
    return keep.stack()
