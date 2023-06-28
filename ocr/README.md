# text-detection

## 简介

为了能将一张图像中的多行文本识别出来，可以将该任务分为两步：

1. 检测图像中每一行文本的位置
2. 根据位置从原始图像截取出一堆子图像
3. 只需识别出子图像的文字，再进行排序组合即可

因此，采用两类模型：

1. 文本检测：CTPN
2. 文本识别：Densenet + ctc

  
## 执行速度

| 图像大小 | 处理器    | 文本行数量 | 速度  |
| -------- | --------- | ---------- | ----- |
| 500kb    | 1070ti    | 20         | 420ms |
| 500kb    | Tesla k80 | 20         | 1s    |


## 训练

### 数据集说明

- CTPN 训练使用的数据集格式与VOC数据集格式相同，目录格式如下：

    ```json
    - VOCdevkit
        - VOC2007
            - Annotations
            - ImageSets
            - JPEGImages
    ```

- Densenet + ctc 使用的数据集分为3部分

  - 文字图像
  - 标注文件：包括图像路径与所对应的文本标记（train.txt, test.txt)
  - 字典文件：包含数据集中的所有文字 (char_std_5990.txt)
  
数据集链接：

- ctpn: 链接: https://pan.baidu.com/s/19iMHzjvNfQS22NdFjZ_2XQ 提取码: nw7a
- densenet: 链接: https://pan.baidu.com/s/1LT9whsTJx-S48rtRTXw5VA 提取码: rugb

关于创建自己的文本识别数据集，可参考：[https://github.com/Sanster/text_renderer](https://github.com/Sanster/text_renderer)。

### CTPN 训练

ctpn 的训练需要传入2个必要参数：

1. 图像目录位置
2. 标注文件目录位置

<模型配置文件位置> 用于指定模型的一些参数，若不指定，将使用默认配置：

```json
{
  "image_channels": 3,  // 图像通道数
  "vgg_trainable": true, // vgg 模型是否可训练
  "lr": 1e-05   // 初始学习率
}
```

<保存模型训练权重文件位置> 若不指定，会保存到当前目录下的model文件夹

训练情况：

```sh
...

Epoch 17/20
6000/6000 [==============================] - 4036s 673ms/step - loss: 0.0895 - rpn_class_loss: 0.0360 - rpn_regress_loss: 0.0534
Epoch 18/20
6000/6000 [==============================] - 4075s 679ms/step - loss: 0.0857 - rpn_class_loss: 0.0341 - rpn_regress_loss: 0.0516
Epoch 19/20
6000/6000 [==============================] - 4035s 673ms/step - loss: 0.0822 - rpn_class_loss: 0.0324 - rpn_regress_loss: 0.0498
Epoch 20/20
6000/6000 [==============================] - 4165s 694ms/step - loss: 0.0792 - rpn_class_loss: 0.0308 - rpn_regress_loss: 0.0484

```

### Densenet 训练

Densnet 的训练需要4个必要参数：

1. 训练图像位置
2. 字典文件位置
3. 训练文件位置
4. 测试文件位置

<模型配置文件位置> 用于指定模型使用的配置文件路径，若不指定，默认配置如下：

```json
{
  "lr": 0.0005, // 初始学习率
  "num_classes": 5990, // 字典大小
  "image_height": 32,   // 图像高
  "image_channels": 1,  // 图像通道数
  "maxlen": 50,         // 最长文本长度
  "dropout_rate": 0.2,  //  随机失活率
  "weight_decay": 0.0001, // 权重衰减率
  "filters": 64         // 模型第一层的核数量
}
```

<保存模型训练权重文件位置> 若不指定，会保存到当前目录下的model文件夹

训练情况：

```sh
Epoch 3/100
25621/25621 [==============================] - 15856s 619ms/step - loss: 0.1035 - acc: 0.9816 - val_loss: 0.1060 - val_acc: 0.9823
Epoch 4/100
25621/25621 [==============================] - 15651s 611ms/step - loss: 0.0798 - acc: 0.9879 - val_loss: 0.0848 - val_acc: 0.9878
Epoch 5/100
25621/25621 [==============================] - 16510s 644ms/step - loss: 0.0732 - acc: 0.9889 - val_loss: 0.0815 - val_acc: 0.9881
Epoch 6/100
25621/25621 [==============================] - 15621s 610ms/step - loss: 0.0691 - acc: 0.9895 - val_loss: 0.0791 - val_acc: 0.9886
Epoch 7/100
25621/25621 [==============================] - 15782s 616ms/step - loss: 0.0666 - acc: 0.9899 - val_loss: 0.0787 - val_acc: 0.9887
Epoch 8/100
25621/25621 [==============================] - 15560s 607ms/step - loss: 0.0645 - acc: 0.9903 - val_loss: 0.0771 - val_acc: 0.9888
```

