# text-detection

## 简介

为了能将一张图像中的多行文本识别出来，可以将该任务分为两步：

1. 检测图像中每一行文本的位置
2. 根据位置从原始图像截取出一堆子图像
3. 识别出子图像的文字，再进行排序组合即可

因此，采用两类模型：

1. 文本检测：CTPN
2. 文本识别：Densenet + ctc

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

