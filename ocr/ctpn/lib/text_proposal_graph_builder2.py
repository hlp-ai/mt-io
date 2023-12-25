import tensorflow as tf

class TextLineCfg:
    SCALE=600
    MAX_SCALE=1200
    TEXT_PROPOSALS_WIDTH=16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO=0.5
    LINE_MIN_SCORE=0.9
    MAX_HORIZONTAL_GAP=60
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MIN_V_OVERLAPS=0.6
    MIN_SIZE_SIM=0.6

# tf.config.run_functions_eagerly(True)  #

class Graph:
    def __init__(self, graph):
        self.graph = graph

    @tf.function
    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not tf.reduce_any(self.graph[:, index]) and tf.reduce_any(self.graph[index, :]):
                v = index
                sub_graphs.append([v])
                while tf.reduce_any(self.graph[v, :]):
                    v = tf.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

@tf.function
def sub_graphs_connected(graph):
    # sub_graphs = []
    sub_graphs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)  #
    graph_shape = tf.shape(graph)
    num_proposals = graph_shape[0]

    def check(index):
        return not tf.reduce_any(graph[:, index]) and tf.reduce_any(graph[index, :])

    def f1(sub_graphs, start):
        v = start
        # sub_graphs.append([v])
        # sub_graphs = sub_graphs.write(sub_graphs.size(), start)  #
        segment = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        segment = segment.write(segment.size(), v)

        def loop_body2(v, segment):
            v = tf.where(graph[v, :])[0][0]
            v = tf.cast(v, dtype=tf.int32)
            segment = segment.write(segment.size(), v)
            return v, segment

        def loop_cond2(v, segment):
            return tf.reduce_any(graph[v, :])

        v, segment = tf.while_loop(loop_cond2, loop_body2, [v, segment])

        segment_tensor = segment.stack()
        sub_graphs = sub_graphs.write(sub_graphs.size(), segment_tensor)
        return sub_graphs

    def f2(result):
        return result

    def loop_body(start, end, sub_graphs):
        sub_graphs = tf.cond(check(start), lambda: f1(sub_graphs, start), lambda: f2(sub_graphs))
        start = start + 1
        return start, end, sub_graphs

    def loop_cond(start, end, sub_graphs):
        return tf.less(start, end)

    start, end, sub_graphs = tf.while_loop(loop_cond, loop_body, [0, num_proposals, sub_graphs])

    """for index in range(num_proposals):
        if not tf.reduce_any(graph[:, index]) and tf.reduce_any(graph[index, :]):
            v = index
            sub_graphs.append([v])
            while tf.reduce_any(graph[v, :]):
                v = tf.where(graph[v, :])[0][0]
                v = tf.cast(v, dtype=tf.int32)
                sub_graphs[-1].append(v)"""

    return sub_graphs
    # return segment.stack()  #

class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """


    @tf.function
    def get_successions(self, index):
        text_proposals = self.text_proposals
        # text_proposals_shape = tf.shape(text_proposals)
        # num_proposals = text_proposals_shape[0]

        result = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        s = tf.convert_to_tensor(self.text_proposals[index, 0])
        left_start = s + 1
        left_end = tf.minimum(s + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])

        def check(i, target, index):
            return tf.equal(self.text_proposals[i, 0], target) and self.meet_v_iou(i, index)

        def f1(result, x):
            result = result.write(result.size(), x)
            return result

        def f2(result):
            return result

        def loop_body(start, end, result):
            i = 0
            for text_proposal in text_proposals:
                result = tf.cond(check(i, start, index), lambda: f1(result, i), lambda: f2(result))
                i = i + 1
            start = start + 1
            print(start)
            return start, end, result

        def loop_cond(start, end, result):
            return tf.less(start, end) and tf.equal(len(result), 0)

        start, end, result = tf.while_loop(loop_cond, loop_body, [left_start, left_end, result])
        return result.stack()

        """while left_start < left_end:
            target = text_proposals[index, 0]
            i = 0
            for text_proposal in text_proposals:
                result = tf.cond(check(i, target, index), lambda: f1(result, i), lambda: f2(result))
                i = i+1
            left_start = tf.cond(check2(len(result)), lambda: f11(left_start), lambda: f12(left_start))"""


        """left = tf.constant(0)
        while left < tf.constant(3):
            target = text_proposals[left, 0]
            print("target")
            print(target)
            for i in range(text_proposals.shape[0]):
                result = tf.cond(check(text_proposals[i, 0], target), lambda: f1(result, i), lambda: f2(result))
            left = tf.cond(check2(len(result)), lambda: f11(left), lambda: f12(left))
        return result.stack()
        
        
        box = self.text_proposals[index]
        results = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        s = tf.convert_to_tensor(box[0])
        left_start = s + 1
        left_end = tf.minimum(s + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])
        
        for i in range(24):
            if self.boxes_table.read(i).shape[0] == None:
                print("None")
            else:
                print(self.boxes_table.read(i))"""


    @tf.function
    def get_precursors(self, index):
        text_proposals = self.text_proposals

        result = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        s = tf.convert_to_tensor(tf.gather_nd(self.text_proposals, tf.concat([index, tf.constant([0])], axis=0)))
        left_end = tf.maximum(s - TextLineCfg.MAX_HORIZONTAL_GAP - 1, 0)
        left_start = s - 1

        def check(i, target, index):
            return tf.equal(self.text_proposals[i, 0], target) and self.meet_v_iou_tf(i, index)

        def f1(result, x):
            print("x")
            print(x)
            result = result.write(result.size(), x)
            return result

        def f2(result):
            return result

        def loop_body(start, end, result):
            i = 0
            for text_proposal in text_proposals:
                result = tf.cond(check(i, start, index), lambda: f1(result, i), lambda: f2(result))
                i = i + 1
            start = start - 1
            return start, end, result

        def loop_cond(start, end, result):
            # return tf.less(end, start)  #
            return tf.less(end, start) and tf.equal(len(result), 0)

        start, end, result = tf.while_loop(loop_cond, loop_body, [left_start, left_end, result])
        return result.stack()

    @tf.function
    def is_succession_node(self, start, succession_index):
        precursors = self.get_precursors(succession_index)
        # return precursors
        return tf.gather(self.scores, start) >= tf.reduce_max(tf.gather(self.scores, precursors))

    @tf.function
    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = tf.maximum(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = tf.minimum(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return tf.maximum(0, y1 - y0 + 1) / tf.minimum(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return tf.minimum(h1, h2) / tf.maximum(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    @tf.function
    def meet_v_iou_tf(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = tf.gather(self.heights, index2)
            a1 = tf.gather_nd(self.text_proposals, tf.concat([index2, tf.constant([1])], axis=0))
            a2 = tf.gather_nd(self.text_proposals, tf.concat([index2, tf.constant([3])], axis=0))
            y0 = tf.maximum(a1, self.text_proposals[index1][1])
            y1 = tf.minimum(a2, self.text_proposals[index1][3])
            return tf.maximum(0, y1 - y0 + 1) / tf.minimum(h1, h2)

        def size_similarity(index1, index2):
            h1 = tf.gather(self.heights, index1)
            h2 = tf.gather(self.heights, index2)
            return tf.minimum(h1, h2) / tf.maximum(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    @tf.function
    def build_graph(self, text_proposals, scores, im_size):
        text_proposals_shape = tf.shape(text_proposals)
        num_proposals = text_proposals_shape[0]
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
        # graph = tf.zeros((num_proposals, num_proposals), tf.bool)


        left_start = 0
        left_end = num_proposals
        # result = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)  #
        result = tf.zeros((num_proposals, num_proposals), tf.bool)

        def check2(succession_index, start):
            return self.is_succession_node(start, succession_index)

        def f3(succession_index, start, result):
            a = tf.expand_dims(tf.convert_to_tensor(start), axis=0)
            pair = tf.concat([a, succession_index], axis=0)
            indices = tf.expand_dims(pair, axis=0)
            result = tf.tensor_scatter_nd_update(result, indices, tf.constant([True]))
            return result

        def f4(succession_index, start, result):
            return result

        def check(successions):
            return tf.equal(tf.size(successions), 0)

        def f1(successions, start, result):
            return result

        def f2(successions, start, result):
            # result = result.write(result.size(), start)  #  for test
            succession_index = tf.gather(successions, tf.argmax(tf.gather(scores, successions)))
            result = tf.cond(check2(succession_index, start), lambda: f3(succession_index, start, result),
                    lambda: f4(succession_index, start, result))

            return result

        def loop_body(start, end, result):
            successions = self.get_successions(start)
            result = tf.cond(check(successions), lambda: f1(successions, start, result), lambda: f2(successions, start, result))
            # s_index = tf.gather(successions, tf.argmax(tf.gather(scores, successions)))
            start = start + 1
            return start, end, result

        def loop_cond(start, end, result):
            return tf.less(tf.convert_to_tensor(start), end)
        start, end, result = tf.while_loop(loop_cond, loop_body, [left_start, left_end, result])

        graph = result

        return graph
        """def text_proposals_loop(text_proposals, scores2):
            def check2(succession_index, index):
                return self.is_succession_node(index, succession_index)

            def f3(succession_index, index, ans):
                ans.append((succession_index, index))
                return 0

            def f4(succession_index, index):
                return 0

            def check(successions):
                return tf.equal(tf.size(successions), 0)

            def f1(succession_index, index, ans):
                tf.cond(check2(succession_index, index), lambda: f3(succession_index, index, ans),
                        lambda: f4(succession_index, index))
                return 1

            def f2(successions, succession_index):
                return 1

            ans = []
            index = 0
            scores2 = tf.identity(scores2, name='scores2')

            for text_proposal in text_proposals:
                successions = self.get_successions(index)
                successions = tf.identity(successions, name='successions')
                s_index = tf.gather(successions, tf.argmax(tf.gather(scores2, successions)))

                tf.cond(check(successions), lambda: f1(s_index, index, ans), lambda: f2(successions, index))
                index = index + 1

            return ans
        ans = text_proposals_loop(text_proposals, scores)"""



        """successions = self.get_successions(index)
            print("test:  #####")
            print(index)
            print(successions)
            return successions"""
            # successions = tf.gather(indices, successions)
        """for index in tf.range(text_proposals):
            box = text_proposals[index]
            left = tf.cast(box[0], tf.int32)
            if boxes_table.read(left).shape[0] == None:
                print("None")
                continue
            indices = boxes_table.read(left)
            successions = tf.gather(indices, self.get_successions(index))
            if tf.size(successions) == 0:
                continue
            succession_index = successions[tf.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph = tf.tensor_scatter_nd_update(graph, [[index, succession_index]], [True])"""

