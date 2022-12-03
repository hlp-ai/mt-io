import os

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from dlocr.custom import LRScheduler, SingleModelCK
from dlocr.densenet import DenseNetOCR

from dlocr.densenet import default_densenet_config_path
from dlocr.densenet.data_reader import OCRDataset
from dlocr.utils import load_config

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("-bs", "--batch_size", help="小批量处理大小", default=64, type=int)
    parser.add_argument("--epochs", help="the number of epochs", default=60, type=int)
    parser.add_argument("--dict_file_path", help="字典文件位置",
                        default="../dictionary/char_std_5991.txt")
    parser.add_argument("--train_file_path", help="tfrecord file for training set",
                        default="500k-sp-train.tfrecord")
    parser.add_argument("--test_file_path", help="tfrecord file for dev set",
                        default="500k-sp-dev.tfrecord")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_densenet_config_path)
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=r'model/weights-densenet.hdf5')
    parser.add_argument("--save_weights_file_path", help="保存模型训练权重文件位置",
                        default=r'model/weights-densenet.hdf5')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    initial_epoch = args.initial_epoch

    config = load_config(args.config_file_path)
    weights_file_path = args.weights_file_path

    if os.path.exists(weights_file_path):
        config["weight_path"] = weights_file_path

    dict_file_path = args.dict_file_path
    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    save_weights_file_path = args.save_weights_file_path

    # TODO: mkdir for save_weight_file_path, not fixed
    if not os.path.exists("model"):
        os.makedirs("model")

    train_data = OCRDataset(dict_file_path, train_file_path, max_label_len=20)
    ds_train = train_data.get_ds(batch_size=batch_size, prefetch_size=51200)

    dev_data = OCRDataset(dict_file_path, test_file_path, max_label_len=20)
    ds_dev = dev_data.get_ds(batch_size=batch_size, prefetch_size=3200)

    ocr = DenseNetOCR(**config)
    ocr.parallel_model.summary()

    checkpoint = SingleModelCK(save_weights_file_path,
                               model=ocr.train_model,
                               save_weights_only=True,
                               save_best_only=True,
                               verbose=1)

    earlystop = EarlyStopping(patience=3, verbose=1)
    # log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=train_data_loader.batch_size,
    #                   write_graph=True,
    #                   write_grads=False)
    #
    # 观测ctc损失的值，一旦损失回升，将学习率缩小一半
    lr_scheduler = LRScheduler(lambda _, lr: lr / 2, watch="loss", watch_his_len=2)

    ocr.parallel_model.fit(ds_train, epochs=epochs, validation_data=ds_dev, callbacks=[checkpoint, earlystop, lr_scheduler])

