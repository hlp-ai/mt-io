import os

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from ocr.custom import LRScheduler, SingleModelCK
from ocr.densenet import get_model
from ocr.densenet.data_reader import OCRDataset, load_dict_sp

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help="小批量处理大小", default=64, type=int)
    parser.add_argument("--epochs", help="the number of epochs", default=60, type=int)
    parser.add_argument("--dict_file_path", help="字典文件位置",
                        default="../dictionary/char_std_5991.txt")
    parser.add_argument("--train_file_path", help="tfrecord file for training set",
                        default="500k-sp-train.tfrecord")
    parser.add_argument("--test_file_path", help="tfrecord file for dev set",
                        default="500k-sp-dev.tfrecord")
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=r'model/weights-densenet.hdf5')
    parser.add_argument("--save_weights_file_path", help="保存模型训练权重文件位置",
                        default=r'model/weights-densenet.hdf5')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    weights_file_path = args.weights_file_path

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

    id_to_char = load_dict_sp(dict_file_path, "UTF-8")
    _, train_model = get_model(num_classes=len(id_to_char))
    train_model.load_weights(weights_file_path)

    train_model.summary()

    checkpoint = SingleModelCK(save_weights_file_path,
                               model=train_model,
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

    train_model.fit(ds_train, epochs=epochs, validation_data=ds_dev, callbacks=[checkpoint, earlystop, lr_scheduler])

