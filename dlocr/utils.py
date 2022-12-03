import json
import cv2


def load_config(config_path):
    with open(config_path, "r") as infile:
        return dict(json.load(infile))


def draw_rect(rect, img):
    cv2.line(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
    cv2.line(img, (rect[2], rect[3]), (rect[6], rect[7]), (255, 0, 0), 2)
    cv2.line(img, (rect[6], rect[7]), (rect[4], rect[5]), (255, 0, 0), 2)
    cv2.line(img, (rect[4], rect[5]), (rect[0], rect[1]), (255, 0, 0), 2)
