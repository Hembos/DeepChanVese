from keras.layers import Input, Lambda, Concatenate, Layer
from tensorflow import float32, Variable
from keras.backend import shape

import numpy as np
import math

import utils
from resFpn import create_res_fpn_model
from rpn import create_rpn_model
from roi import ROI_Layer

class Model():
    def __init__(self, isTrainig, config) -> None:
        self.isTraining = isTrainig
        self.config = config
        self.anchors_generated = False
        self.anchors = []

        self.model = self.build()

    def build(self):
        height = int(self.config["IMAGE_SHAPE"]["height"])
        width = int(self.config["IMAGE_SHAPE"]["width"])
        channels = int(self.config["IMAGE_SHAPE"]["channels"])

        input_img = Input([None, None, channels], name="img_input")
        input_image_meta = Input(shape=[13], name="input_image_meta")

        if self.isTraining:
            input_rpn_bbox = Input([None, 4], name="input_rpn_bbox", dtype=float32)

            input_gt_bbox = Input([None, 4], name="input_gt_bbox", dtype=float32)

            # gt_boxes = Lambda(lambda x: utils.normalize_boxes(x, shape(input_img)[1:3]))(input_gt_bbox)
            gt_boxes = Lambda(lambda x: utils.normalize_boxes(x))([input_gt_bbox, input_img])
            
            input_gt_masks = Input(shape=[height, width, None], 
                                   name="input_gt_masks", dtype=bool)
            
        top_down_pyramid_size = int(self.config["FPN"]["top_down_pyramid_size"])
        p2, p3, p4, p5, p6 = create_res_fpn_model(input_img, top_down_pyramid_size)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        anchor_scales = [int(x) for x in self.config["RPN"]["anchor_scales"].split(",")]
        anchor_ratios = [float(x) for x in self.config["RPN"]["anchor_ratios"].split(",")]
        feature_strides = [int(x) for x in self.config["FPN"]["feature_strides"].split(",")]
        anchor_stride = int(self.config["RPN"]["anchor_stride"])
        if self.isTraining:
            feature_shapes = np.array([[int(math.ceil(1024 / stride)),
                                        int(math.ceil(1024 / stride))]
                                        for stride in feature_strides])
            
            if not self.anchors_generated:
                self.anchors = utils.generate_pyramid_anchors(anchor_scales, anchor_ratios, feature_shapes, feature_strides, anchor_stride)
                self.anchors = np.broadcast_to(self.anchors, (int(self.config["TRAINING"]["batch_size"]),) + self.anchors.shape)
                self.anchors = Lambda(lambda x: Variable(self.anchors), name="anchors")(input_img)

        rpn = create_rpn_model(anchor_stride, len(anchor_ratios), top_down_pyramid_size)

        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]
        
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        max_proposals_num = int(self.config["ROI"]["max_rois_training"]) if self.isTraining else int(self.config["ROI"]["max_rois_detect"])
        rpn_rois = ROI_Layer(max_proposals_num, float(self.config["ROI"]["nms_threshold"]), 
                      np.array([float(x) for x in self.config["RPN"]["rpn_bbox_std_dev"].split(",")]),
                      int(self.config["ROI"]["pre_nms_limit"]),
                      int(self.config["TRAINING"]["batch_size"]),
                      name="ROI")([rpn_class, rpn_bbox, self.anchors])
        
        if self.isTraining:
            active_class_ids = Lambda(lambda x: utils.parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            target_rois = rpn_rois




    def compile():
        pass

    def train():
        pass

    def detect():
        pass

class DetectionTargetLayer(Layer):
    def __init__(self, batch_size, train_rois_per_image, mask_width, mask_height, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.train_rois_per_image = train_rois_per_image
        self.mask_width = mask_width
        self.mask_height = mask_height

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.batch_size, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 4),  # rois
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 4),  # deltas
            (None, self.train_rois_per_image, self.mask_height,
             self.mask_width)  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
    
def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks

if __name__=="__main__":
    from configparser import ConfigParser

    config = ConfigParser()
    config.read("C:\\Users\\Artur\\Documents\\DiplomProjects\\DeepChanVese\\settings.ini")
    model = Model(isTrainig=True, config=config)
