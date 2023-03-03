# -*- coding: utf-8 -*-
import argparse
import time

from dlocr.ocr import OCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path_pattern", help="图像位置模式", required=True)
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--ctpn_weight_path", help="ctpn模型权重文件位置", required=True)
    parser.add_argument("--densenet_weight_path", help="densenet模型权重文件位置", required=True)
    parser.add_argument("--adjust", help="是否对图像进行适当裁剪",
                        default=True, type=bool)

    args = parser.parse_args()

    app = OCR(ctpn_weight_path=args.ctpn_weight_path,
                           densenet_weight_path=args.densenet_weight_path,
                           dict_path=args.dict_file_path)

    start_no = 16
    end_no = 27
    img_path_pattern = args.image_path_pattern
    for i in range(start_no, end_no):
        img_fn = img_path_pattern.format(i)
        start_time = time.time()
        _, texts = app.detect(img_fn, args.adjust)
        print("cost", (time.time() - start_time) * 1000, "ms")
        texts = '\n'.join(texts)
        print(texts)
        with open(img_fn+".txt", "w", encoding="utf-8") as f:
            f.write(texts)
