import os

from keras.callbacks import EarlyStopping, ModelCheckpoint

from ocr.ctpn.data_reader import get_ctpn_ds
from ocr.ctpn.train import _rpn_loss_regr, _rpn_loss_cls
from tensorflow.keras.optimizers import Adam

from ocr.ctpn.data_loader import DataLoader
from ocr.ctpn.model import get_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="D:/dataset/ocr/VOCdevkit/VOC2007", help="训练数据位置")
    parser.add_argument("--weights_file_path", default="./ctpn_weights.hdf5", help="模型权重文件位置")
    parser.add_argument("--vgg16_weights_path",
                        default="../weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                        help="VGG16权重文件路径")
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)

    args = parser.parse_args()

    save_path = args.weights_file_path
    model = get_model(vgg_weights_path=args.vgg16_weights_path)
    model.summary()
    if os.path.exists(save_path):
        print("Loading model for training...")
        model.load_weights(save_path)

    print("Loading training data...")
    data_path = args.data_path
    ds = get_ctpn_ds(data_path)

    model.compile(optimizer=Adam(1e-05),
                        loss={'rpn_regress': _rpn_loss_regr, 'rpn_class': _rpn_loss_cls},
                        loss_weights={'rpn_regress': 1.0, 'rpn_class': 1.0})

    checkpoint = ModelCheckpoint(save_path, model=model, save_weights_only=True, monitor='loss', save_best_only=True)
    earlystop = EarlyStopping(patience=2, monitor='loss')
    # lr_scheduler = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
    #                              initial_epoch=args.initial_epoch,
    #                              steps_per_epoch=data_loader.steps_per_epoch,
    #                              cycle_length=8,
    #                              lr_decay=0.5,
    #                              mult_factor=1.2)

    model.fit(ds, epochs=1,callbacks=[checkpoint, earlystop])
