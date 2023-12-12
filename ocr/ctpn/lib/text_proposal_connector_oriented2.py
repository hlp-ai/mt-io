import numpy as np
import tensorflow as tf
from .text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnectorOriented:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # 判断X是否只包含一个点
        X = tf.cast(X, dtype=tf.float32)
        Y = tf.cast(Y, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        x2 = tf.cast(x2, dtype=tf.float32)
        condition = tf.reduce_all(tf.equal(X, X[0]))
        if tf.reduce_sum(tf.cast(condition, tf.int32)) == tf.size(X):
            return Y[0], Y[0]

        A = tf.stack([X, tf.ones_like(X)], axis=1)
        b = tf.expand_dims(Y, axis=1)
        # 使用最小二乘法进行拟合
        k, b = tf.linalg.lstsq(A, b, fast=False)
        return x1*k+b, x2*k+b


    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals: boxes
        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = tf.zeros((len(tp_groups), 8), tf.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = tf.gather(text_proposals, tp_indices)  # 每个文本行的全部小框

            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            A = tf.stack([X, tf.ones_like(X)], axis=1)
            b = tf.expand_dims(Y, axis=1)
            # 使用最小二乘法进行拟合
            z1 = tf.linalg.lstsq(A, b, fast=False)
            z1 = tf.squeeze(z1, axis=1)
            x0 = tf.reduce_min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = tf.reduce_max(text_line_boxes[:, 2])  # 文本行x坐标最大值
            offset = tf.cast((text_line_boxes[0, 2] - text_line_boxes[0, 0]), tf.float32) * 0.5  # 小框宽度的一半
            # 多项式拟合
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], tf.cast(x0, tf.float32) + offset, tf.cast(x1, tf.float32) - offset)
            lb_y, rb_y = self.fit_y(X, text_line_boxes[:, 3], x0 + tf.cast(offset, tf.int32), x1 - tf.cast(offset, tf.int32))
            score = tf.reduce_sum(tf.gather(scores, tp_indices)) / tf.cast(len(tp_indices),
                                                                        tf.float32)  # 求全部小框得分的均值作为文本行的均值
            x0 = tf.cast(x0, tf.float32)
            a0 = x0.numpy().sum()
            a1 = tf.minimum(lt_y, rt_y).numpy().sum()
            a2 = tf.cast(x1, tf.float32).numpy().sum()
            a3 = tf.maximum(lb_y, rb_y).numpy().sum()
            a4 = score.numpy().sum()
            a5 = z1[0].numpy().sum()
            a6 = z1[1].numpy().sum()
            height = tf.reduce_mean(text_line_boxes[:, 3] - text_line_boxes[:, 1])
            a7 = tf.add(tf.cast(height, tf.float32), tf.constant(2.5, dtype=tf.float32)).numpy().sum()
            # 创建更新值的张量
            update_value = tf.constant([a0, a1, a2, a3, a4, a5, a6, a7], dtype=tf.float32)

            # 创建索引张量
            indices = tf.constant([[index, 0], [index, 1], [index, 2], [index, 3], [index, 4], [index, 5], [index, 6], [index, 7]])

            # 使用 tf.tensor_scatter_nd_update() 函数进行更新
            text_lines = tf.tensor_scatter_nd_update(text_lines, indices, update_value)

        text_recs = tf.zeros((text_lines.shape[0], 9), dtype=tf.int32)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1
            x2 = line[2]
            y2 = line[5] * line[2] + b1
            x3 = line[0]
            y3 = line[5] * line[0] + b2
            x4 = line[2]
            y4 = line[5] * line[2] + b2
            disX = x2 - x1
            disY = y2 - y1
            width = tf.sqrt(tf.add(disX * disX, disY * disY))
            fTmp0 = y3 - y1
            fTmp1 = fTmp0 * disY / width
            x = tf.abs(fTmp1 * disX / width)
            y = tf.abs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y

            a0 = tf.cast(x1, dtype=tf.int32).numpy().sum()
            a1 = tf.cast(y1, dtype=tf.int32).numpy().sum()
            a2 = tf.cast(x2, dtype=tf.int32).numpy().sum()
            a3 = tf.cast(y2, dtype=tf.int32).numpy().sum()
            a4 = tf.cast(x3, dtype=tf.int32).numpy().sum()
            a5 = tf.cast(y3, dtype=tf.int32).numpy().sum()
            a6 = tf.cast(x4, dtype=tf.int32).numpy().sum()
            a7 = tf.cast(y4, dtype=tf.int32).numpy().sum()
            a8 = tf.cast(line[4], dtype=tf.int32).numpy().sum()
            indices = tf.constant([[index, 0], [index, 1], [index, 2], [index, 3], [index, 4], [index, 5], [index, 6], [index, 7], [index, 8]])
            update_value = tf.constant([a0, a1, a2, a3, a4, a5, a6, a7, a8], dtype=tf.int32)
            text_recs = tf.tensor_scatter_nd_update(text_recs, indices, update_value)
            index = index + 1

        return text_recs