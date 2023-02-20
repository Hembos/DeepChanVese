from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Add, Input, Resizing, Lambda
import tensorflow as tf
from keras import Model

def create_rpn_model(anchor_stride, num_anchors_per_location, depth):
    input_feature_map = Input(shape=[None, None, depth], name="input_feature_map_rpn")

    shared = Conv2D(512, (3, 3), padding="same", activation="relu", strides=anchor_stride, name="rpn_conv1")(input_feature_map)
    x = Conv2D(2 * num_anchors_per_location, (1, 1), padding="valid", activation="linear", name="rpn_class_score")(shared)
    rpn_class_logits = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    x = Conv2D(num_anchors_per_location * 4, (1, 1), padding="valid", activation="linear", name="rpn_bbox_pred")(shared)

    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return Model([input_feature_map], [rpn_class_logits, rpn_probs, rpn_bbox], name="rpn_model")
