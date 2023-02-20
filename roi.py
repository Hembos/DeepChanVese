from keras.layers import Layer
import numpy as np
from tensorflow import minimum, shape
import tensorflow as tf

from utils import apply_func, apply_box_deltas_graph, clip_boxes_graph

class ROI_Layer(Layer):
    def __init__(self, max_num_proposals, nms_threshold, rpn_bbox_std_dev, pre_nms_limit, batch_size, **kwargs):
        super(ROI_Layer, self).__init__(**kwargs)

        self.max_num_proposals = max_num_proposals
        self.nms_threshold = nms_threshold
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.pre_nms_limit = pre_nms_limit
        self.batch_size = batch_size

    def call(self, inputs):
        scores = inputs[0][:,:,1]

        deltas = inputs[1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])

        anchors = inputs[2]

        pre_nms_limit = minimum(self.pre_nms_limit, shape(anchors)[1])
        ix = tf.math.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        scores = apply_func([scores, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        deltas = apply_func([deltas, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        pre_nms_anchors = apply_func([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.batch_size, names=["pre_nms_anchors"])
        
        boxes = apply_func([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                            self.batch_size, names=["refined_anchors"])

        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = apply_func(boxes, lambda x: clip_boxes_graph(x, window), 
                           self.batch_size, names=["refined_anchors_clipped"])

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        
        proposals = apply_func([boxes, scores], nms, self.batch_size)

        return proposals
    
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)