from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from dlocr.ctpn import CTPN
from dlocr.ctpn import default_ctpn_config_path
from dlocr.ctpn.data_loader import DataLoader
from dlocr.custom import SingleModelCK
from dlocr.utils import load_config

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("--epochs", help="迭代数", default=20, type=int)
    parser.add_argument("--images_dir", help="图像位置", default="E:\data\VOCdevkit\VOC2007\JPEGImages")
    parser.add_argument("--anno_dir", help="标注文件位置", default="E:\data\VOCdevkit\VOC2007\Annotations")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=None)
    parser.add_argument("--save_weights_file_path", help="保存模型训练权重文件位置",
                        default=r'./model/weights-ctpnlstm-{epoch:02d}.hdf5')

    args = parser.parse_args()

    config = load_config(args.config_file_path)

    weights_file_path = args.weights_file_path
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path

    ctpn = CTPN(**config)

    save_weigths_file_path = args.save_weights_file_path

    data_loader = DataLoader(args.anno_dir, args.images_dir)

    checkpoint = SingleModelCK(save_weigths_file_path, model=ctpn.train_model, save_weights_only=True, monitor='loss')
    # checkpoint = ModelCheckpoint(save_weigths_file_path, save_weights_only=True)
    earlystop = EarlyStopping(patience=3, monitor='loss')
    # log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False)
    log = TensorBoard(log_dir="./logs", update_freq=500)

    ctpn.train(data_loader.load_data(),
               epochs=args.epochs,
               steps_per_epoch=data_loader.steps_per_epoch,
               callbacks=[checkpoint, earlystop],
               initial_epoch=args.initial_epoch)
