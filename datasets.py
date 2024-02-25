import math
import numpy as np
from PIL import Image
from tensorflow import keras
from random import sample, shuffle


class YoloDatasets(keras.utils.Sequence):
    def __init__(self, object_lines, mask_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                 epoch_now, num_epochs, mosaic=True):
        self.object_lines = object_lines
        self.mask_lines = mask_lines
        self.length = len(self.object_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors_mask = anchors_mask
        self.epoch_now = epoch_now - 1
        self.num_epochs = num_epochs
        self.mosaic = mosaic
        self.threshold = 4
        self.mosaic_ratio = 0.7

    def __len__(self):
        return math.ceil(len(self.object_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data = []
        box_data = []
        mask_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length

            if self.mosaic:
                if self.rand() < 0.5 and self.epoch_now < self.num_epochs * self.mosaic_ratio:
                    object_lines = sample(self.object_lines, 3)
                    object_lines.append(self.object_lines[i])
                    shuffle(object_lines)
                    mask_lines = sample(self.mask_lines, 3)
                    mask_lines.append(self.mask_lines[i])
                    shuffle(mask_lines)
                    image, box, mask = self.get_Mosaic_data(self.object_lines[i], self.mask_lines[i], self.input_shape)
                else:
                    image, box, mask = self.get_random_data(self.object_lines[i], self.mask_lines[i], self.input_shape)
            else:
                image, box, mask = self.get_random_data(self.object_lines[i], self.mask_lines[i], self.input_shape)

            image_data.append(image)
            box_data.append(box)
            mask_data.append(mask)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        mask_data = np.array(mask_data)
        box_true = self.preprocess_truth(box_data, self.input_shape, self.anchors, self.num_classes)
        truth = [box_true, mask_data]
        return [image_data, *truth], np.zeros(self.batch_size)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_Mosaic_data(self, object_lines, mask_lines, input_shape, jitter=0.3, max_boxes=500):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        mask_datas = []
        box_datas = []
        index = 0
        for _ in object_lines:
            object_line = object_lines.split()
            mask_line = mask_lines.split("\\")[-1].split(".")[0] + '.png'
            image = Image.open(object_line[0])
            iw, ih = image.size

            box = np.array([np.array(list(map(int, box.split(',')))) for box in object_line[1:]])
            mask = Image.open(mask_line)

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            mask = mask.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            new_mask = Image.new('L', (w, h), (128))
            new_mask.paste(mask, (dx, dy))
            mask_data = np.array(new_mask)

            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            mask_datas.append(mask_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
        new_image = np.array(new_image, np.float32)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes

        new_mask = np.zeros([h, w])
        new_mask[:cuty, :cutx] = mask_datas[0][:cuty, :cutx]
        new_mask[cuty:, :cutx] = mask_datas[1][cuty:, :cutx]
        new_mask[cuty:, cutx:] = mask_datas[2][cuty:, cutx:]
        new_mask[:cuty, cutx:] = mask_datas[3][:cuty, cutx:]
        new_mask = np.array(new_mask, np.float32)
        new_mask = np.expand_dims(new_mask, axis=-1)

        return new_image, box_data, new_mask

    def get_random_data(self, object_lines, mask_lines, input_shape, max_boxes=500):
        object_line = object_lines.split()
        mask_line = mask_lines.split("\\")[-1].split(".")[0] + '.png'
        image = Image.open(object_line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in object_line[1:]])
        mask = Image.open(mask_line)

        image_data = np.array(image, np.float32)
        image_data /= 255.0

        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            box_data[:len(box)] = box

        # 对Mask进行调整
        mask_data = np.array(mask, np.float32)
        mask_data = np.expand_dims(mask_data, axis=-1)

        return image_data, box_data, mask_data

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def preprocess_truth(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4] < num_classes).all()
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        num_layers = len(self.anchors_mask)
        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]
        box_best_ratios = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l])),
                                    dtype='float32') for l in range(num_layers)]
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        anchors = np.array(anchors, np.float32)
        valid_mask = boxes_wh[..., 0] > 0
        for b in range(m):
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            ratios_of_gt_anchors = np.expand_dims(wh, 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(wh, 1)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            max_ratios = np.max(ratios, axis=-1)
            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for l in range(num_layers):
                    for k, n in enumerate(self.anchors_mask[l]):
                        if not over_threshold[n]:
                            continue
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        offsets = self.get_near_points(true_boxes[b, t, 0] * grid_shapes[l][1],
                                                       true_boxes[b, t, 1] * grid_shapes[l][0], i, j)
                        for offset in offsets:
                            local_i = i + offset[0]
                            local_j = j + offset[1]

                            if local_i >= grid_shapes[l][1] or local_i < 0 or local_j >= grid_shapes[l][0] or local_j < 0:
                                continue

                            if box_best_ratios[l][b, local_j, local_i, k] != 0:
                                if box_best_ratios[l][b, local_j, local_i, k] > ratio[n]:
                                    y_true[l][b, local_j, local_i, k, :] = 0
                                else:
                                    continue
                            c = true_boxes[b, t, 4].astype('int32')
                            y_true[l][b, local_j, local_i, k, 0:4] = true_boxes[b, t, 0:4]
                            y_true[l][b, local_j, local_i, k, 4] = 1
                            y_true[l][b, local_j, local_i, k, 5 + c] = 1
                            box_best_ratios[l][b, local_j, local_i, k] = ratio[n]
        return y_true
