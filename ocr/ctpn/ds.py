"""Data processing for VOC dataset"""
import os
from glob import glob

import xmltodict
from tqdm import tqdm


def get_annotation_files(annotations_dir):
    return glob(annotations_dir + '/*.xml')


def get_annotation(annotation_file, images_dir):
    gtboxes = []
    with open(annotation_file, 'rb') as f:
        xml = xmltodict.parse(f)
        imgfile = xml['annotation']['filename']
        bboxes = xml['annotation']['object']
        if (type(bboxes) != list):
            x1 = bboxes['bndbox']['xmin']
            y1 = bboxes['bndbox']['ymin']
            x2 = bboxes['bndbox']['xmax']
            y2 = bboxes['bndbox']['ymax']
            gtboxes.append((int(x1), int(y1), int(x2), int(y2)))
        else:
            for bbox in bboxes:
                x1, y1, x2, y2 = (bbox['bndbox']['xmin'],
                                  bbox['bndbox']['ymin'],
                                  bbox['bndbox']['xmax'],
                                  bbox['bndbox']['ymax'])
                gtboxes.append((int(x1), int(y1), int(x2), int(y2)))

    return gtboxes, os.path.join(images_dir, imgfile)


def gen_annotation_data(annotations_dir, images_dir):
    annotation_files = get_annotation_files(annotations_dir)
    for f in annotation_files:
        gtboxes, imgfile = get_annotation(f, images_dir)
        yield {"img_file": imgfile, "bboxes": gtboxes}


def get_annotation_data(annotations_dir, images_dir):
    samples = []
    annotation_files = get_annotation_files(annotations_dir)
    for f in tqdm(annotation_files, desc="Parsing annotation file"):
        gtboxes, imgfile = get_annotation(f, images_dir)
        samples.append({"img_file": imgfile, "bboxes": gtboxes})
    return samples


if __name__ == "__main__":
    anno_dir = r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations"
    images_dir = r"D:\dataset\ocr\VOCdevkit\VOC2007\JPEGImages"

    # xml_files = get_annotation_files(r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations")

    # print(len(xml_files))
    # print(xml_files)
    #
    # xmlpath = r"D:\dataset\ocr\VOCdevkit\VOC2007\Annotations\img_1001.xml"
    #
    # gtboxes, imgfile = readxml(xmlpath, r"D:\dataset\ocr\VOCdevkit\VOC2007\JPEGImages")
    # print(imgfile)
    # print(gtboxes)

    # for s in gen_annotation_data(anno_dir, images_dir):
    #     print(s)

    samples = get_annotation_data(anno_dir, images_dir)
    print(len(samples))
