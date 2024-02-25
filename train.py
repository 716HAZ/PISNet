import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard

from datasets import YoloDatasets
from model import get_train_model, yolo_body
from model_checkpoint import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(path):
    with open(path, encoding='utf-8') as f:
        anchor = f.readline()
    anchor = [float(x) for x in anchor.split(',')]
    anchor = np.array(anchor).reshape(-1, 2)
    return anchor, len(anchor)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    input_shape = [256, 512]
    lr = 1e-4
    batch_size = 2
    init_epoch = 0
    num_epochs = 120
    create_dir("files")
    model_path = 'files/model_best.h5'
    classes_path = 'new_data/voc_classes.txt'
    anchors_path = 'new_data/yolo_anchors.txt'
    train_object_path = 'new_data/2007_train.txt'
    val_object_path = 'new_data/2007_val.txt'
    train_mask_path = 'new_data/2007_mask_train.txt'
    val_mask_path = 'new_data/2007_mask_val.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    with open(train_object_path, encoding='utf-8') as f:
        train_object_lines = f.readlines()
    with open(val_object_path, encoding='utf-8') as f:
        val_object_lines = f.readlines()
    with open(train_mask_path, encoding='utf-8') as f:
        train_mask_lines = f.readlines()
    with open(val_mask_path, encoding='utf-8') as f:
        val_mask_lines = f.readlines()
    num_train = len(train_object_lines)
    num_val = len(val_object_lines)
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    train_dataset = YoloDatasets(train_object_lines, train_mask_lines, input_shape, anchors, batch_size, num_classes,
                                 anchors_mask, 0, num_epochs, mosaic=False)
    val_dataset = YoloDatasets(val_object_lines, val_mask_lines, input_shape, anchors, batch_size, num_classes,
                               anchors_mask, 0, num_epochs, mosaic=False)

    model_body = yolo_body([input_shape[0], input_shape[1], 3], anchors_mask, num_classes)
    model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)
    model.compile(optimizer=Adam(lr=lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    if os.path.exists(model_path):
        print('-------------load the model-----------------')
        model.load_weights(model_path)

    callbacks = [ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7, verbose=1),
                 TensorBoard()]

    model.fit(x=train_dataset, validation_data=val_dataset, epochs=num_epochs, shuffle=True, callbacks=callbacks)
