import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse

from dnnlib.retinanet import model

class ObjectDetector():
    r"""Interface of RetinaNet for inference on a single RGB image.

    Parameters
    ----------
    weights_path : str
        Path to weights file of the network
    cls_path : str
        Path to CSV file of the classes names
    """
    def __init__(self, weights_path, cls_path):
        # initialize object detector class
        # read classes list
        with open(cls_path, 'r') as f:
            classes = self.load_classes(csv.reader(f, delimiter=','))
        # prepare labels dictionary
        self.labels = {}
        for key, value in classes.items():
            self.labels[value] = key
        # load RetinaNet model
        self.retinanet = model.resnet50(num_classes=len(self.labels))
        self.retinanet.load_state_dict(torch.load(weights_path))
        # copy model to GPU
        if torch.cuda.is_available():
            self.retinanet = self.retinanet.cuda()
        # set model to eval mode
        self.retinanet.training = False
        self.retinanet.eval()

    def load_classes(self, csv_reader):
        # load classes from CSV file
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = int(class_id)

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def draw_caption(self, image, box, caption):
        # draw class name on top of bounding box
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def detect_image(self, image, out_dir):
        # perform object detection using RetinaNet model
        image_orig = image.copy()
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        # perform network inference
        with torch.no_grad():
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()
            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = self.retinanet(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            detections = list()
            # process and visualize detections
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = self.labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                detections.append([label_name, x1, y1, x2, y2, score])
                self.draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(out_dir, 'detect_out.png'), image_orig)
        return detections
