import os

import tensorflow as tf

from ocr.custom import LRScheduler
from ocr.densenet.data_reader import OCRDataset, load_dict_sp
from ocr.densenet.model_with_process import get_model_with_process


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help="小批量处理大小", default=64, type=int)
    parser.add_argument("--epochs", help="the number of epochs", default=60, type=int)
    parser.add_argument("--dict_file_path", help="字典文件位置", required=True)
    parser.add_argument("--train_file_path", help="tfrecord file for training set", required=True)
    parser.add_argument("--test_file_path", help="tfrecord file for dev set", required=True)
    parser.add_argument("--weights_file_path", default="./densenet.hdf5")
    parser.add_argument("--log_dir", default="./logs")

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    dict_file_path = args.dict_file_path
    train_file_path = args.train_file_path
    test_file_path = args.test_file_path

    train_data = OCRDataset(dict_file_path, train_file_path, max_label_len=50)
    ds_train = train_data.get_ds(batch_size=batch_size)

    dev_data = OCRDataset(dict_file_path, test_file_path, max_label_len=50)
    ds_dev = dev_data.get_ds(batch_size=batch_size)

    id_to_char = load_dict_sp(dict_file_path, "UTF-8")
    _, train_model = get_model_with_process(num_classes=len(id_to_char))

    weights_file_path = args.weights_file_path
    if os.path.exists(weights_file_path):
        print("Loading weight from", weights_file_path)
        train_model.load_weights(weights_file_path)

    train_model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file_path,
                               save_weights_only=True,
                               save_best_only=True,
                               verbose=1)

    earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, update_freq=1000)
    #
    # 观测ctc损失的值，一旦损失回升，将学习率缩小一半
    lr_scheduler = LRScheduler(lambda _, lr: lr / 2, watch="loss", watch_his_len=2)

    train_model.fit(ds_train, epochs=epochs, validation_data=ds_dev, callbacks=[checkpoint, earlystop, lr_scheduler, tb_callback])