import math
import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1e-15


def box_ciou(b1, b2):
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)

    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) -
                     tf.math.atan2(b2_wh[..., 0],K.maximum(b2_wh[..., 1], K.epsilon()))) / (math.pi * math.pi)
    alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)

    grid_shape = K.shape(feats)[1:3]
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = (K.sigmoid(feats[..., :2]) * 2 - 0.5 + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = (K.sigmoid(feats[..., 2:4]) * 2) ** 2 * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def total_loss(args, input_shape, anchors, anchors_mask, num_classes, box_ratio=0.05, obj_ratio=1, cls_ratio=0.5):
    balance = [0.4, 1.0, 4]
    num_layers = len(anchors_mask)

    mask_true = args[-1]
    yolo_seg_out = args[num_layers] * mask_true
    seg_loss = dice_loss(mask_true, yolo_seg_out)

    object_true = args[num_layers + 1]
    object_pre_out = args[:num_layers]
    input_shape = K.cast(input_shape, K.dtype(object_true[0]))

    obj_loss = 0
    for l in range(num_layers):
        object_mask = object_true[l][..., 4:5]
        true_class_probs = object_true[l][..., 5:]
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(object_pre_out[l], anchors[anchors_mask[l]],
                                                                  num_classes, input_shape, calc_loss=True)

        pred_box = K.concatenate([pred_xy, pred_wh])
        raw_true_box = object_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * (1 - ciou)
        tobj = tf.where(tf.equal(object_mask, 1), tf.maximum(ciou, tf.zeros_like(ciou)), tf.zeros_like(ciou))
        confidence_loss = K.binary_crossentropy(tobj, raw_pred[..., 4:5], from_logits=True)
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)
        num_pos = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)

        location_loss = K.sum(ciou_loss) * box_ratio / num_pos
        confidence_loss = K.mean(confidence_loss) * balance[l] * obj_ratio
        class_loss = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        obj_loss += location_loss + confidence_loss + class_loss

    loss = seg_loss + obj_loss
    return loss
