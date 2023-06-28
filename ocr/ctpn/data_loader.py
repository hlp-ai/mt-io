import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np

from ocr.ctpn.lib.utils import random_uniform_num, readxml, cal_rpn, IMAGE_MEAN, bbox_transfor_inv


class DataLoader:

    def __init__(self, anno_dir, images_dir, cache_size=64):
        self.anno_dir = anno_dir
        self.images_dir = images_dir
        self.batch_size = 1

        # list xml
        self.xmlfiles = glob(anno_dir + '/*.xml')
        self.total_size = len(self.xmlfiles)
        self.cache_size = cache_size
        self.__rd = random_uniform_num(self.total_size)
        self.__data_queue = []
        self.xmlfiles = np.array(self.xmlfiles)
        self.steps_per_epoch = self.total_size // self.batch_size
        self.__init_queue()

    def __init_queue(self):
        with ThreadPoolExecutor() as executor:
            for data in executor.map(lambda xml_path: self.__single_sample(xml_path),
                                     self.xmlfiles[self.__rd.get(self.cache_size)]):
                self.__data_queue.append(data)

    def __single_sample(self, xml_path):
        gtboxes, imgfile = readxml(xml_path)  # gtbox: (x1,y1,x2,y2), (leftbottom, topright)
        img = cv2.imread(os.path.join(self.images_dir, imgfile))
        return gtboxes, imgfile, img

    def load_data(self):
        while True:
            if len(self.__data_queue) == 0:  # reload data from disk files
                self.__init_queue()

            gtboxes, imgfile, img = self.__data_queue.pop(0)  # remove the first example
            h, w, c = img.shape

            # 随机水平翻转
            if np.random.randint(0, 100) > 50:
                img = img[:, ::-1, :]  # clip image
                # clip x
                newx1 = w - gtboxes[:, 2] - 1
                newx2 = w - gtboxes[:, 0] - 1
                gtboxes[:, 0] = newx1
                gtboxes[:, 2] = newx2

            # 每个锚框的类别, 垂直高度位置偏差
            [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtboxes)

            # 规范化图像
            m_img = img - IMAGE_MEAN
            m_img = np.expand_dims(m_img, axis=0)

            regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

            cls = np.expand_dims(cls, axis=0)
            cls = np.expand_dims(cls, axis=1)
            regr = np.expand_dims(regr, axis=0)

            yield m_img, {'rpn_class': cls, 'rpn_regress': regr}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xmlpath = r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations\img_1001.xml"
    imgpath = r"D:\dataset\ocr\VOCdevkit\VOC2007\JPEGImages\img_1001.jpg"

    gtbox, _ = readxml(xmlpath)
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    print(h, w)

    [cls, regr], base_anchor = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
    print(cls.shape)
    print(cls[:10])
    print(regr.shape)
    print(regr[:10])

    regr = np.expand_dims(regr, axis=0)
    inv_anchor = bbox_transfor_inv(base_anchor, regr)
    anchors = inv_anchor[cls == 1]
    anchors = anchors.astype(int)
    for i in anchors:
        cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 3)
    plt.imshow(img)
    plt.show()
