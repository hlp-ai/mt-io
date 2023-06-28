# -*- coding: utf-8 -*-
import argparse
import time

from ocr.detect import OCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置", required=True)
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--ctpn_weight_path", help="ctpn模型权重文件位置", required=True)
    parser.add_argument("--densenet_weight_path", help="densenet模型权重文件位置", required=True)
    parser.add_argument("--adjust", help="是否对图像进行适当裁剪",
                        default=True, type=bool)

    args = parser.parse_args()

    app = OCR(ctpn_weight_path=args.ctpn_weight_path,
                           densenet_weight_path=args.densenet_weight_path,
                           dict_path=args.dict_file_path)
    start_time = time.time()
    _, texts = app.detect(args.image_path, args.adjust)
    print('\n'.join(texts))
    print("cost", (time.time() - start_time) * 1000, "ms")
