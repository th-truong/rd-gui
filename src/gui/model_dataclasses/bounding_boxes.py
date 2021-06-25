from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as F
import cv2
import matplotlib.cm
import pandas as pd
from pathlib import Path


class BoxDetections():
    # TODO: use dataclass to improve
    def __init__(self, boxes, relations, relations_labels,
                 class_labels, width=None, height=None):
        """[summary]

        Args:
            boxes ([type]): [description]
            width ([type], optional): [description]. Defaults to None.
            height ([type], optional): [description]. Defaults to None.
        """
        self.detections = boxes
        self.relations = relations
        self.width = width
        self.height = height
        self.class_labels = class_labels
        self.relations_labels = relations_labels

    def boxes_as_df(self):
        scores = [x['score'] for x in self.detections]
        labels = [x['id'] for x in self.detections]
        return pd.DataFrame({'Object': labels,
                             'Confidence': scores})

    def relations_as_df(self):
        scores = [x['score'] for x in self.relations]
        relations = [x['person_id'] + " wears " + x['obj_id'] for x in self.relations]
        # TODO: fix this when you need to use multiple relationships
        return pd.DataFrame({'Relationship': relations,
                             'Confidence': scores})

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.detections[key]
        elif isinstance(key, slice):
            return self.detections[key]
        elif isinstance(key, str):
            if key in self.class_labels:
                return [x for x in self.detections if x['label'] == key]
            else:
                return [x for x in self.detections if x['id'] == key]
        elif isinstance(key, float):
            return [x for x in self.detections if x['score'] > key]
        else:
            raise KeyError("The key must be one of type slice, int, str, or float.")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"self.detections: {self.detections}, \
                self.width: {repr(self.width)}, \
                self.height: {repr(self.height)})")

    @classmethod
    def load_model_out(cls, pred, width, height, model_info):
        if model_info['name'] == "relatable_clothing":
            obj_categories = model_info['classes']
            boxes = []
            relations = []
            pred_boxes = pred[0]
            for i, score in enumerate(pred_boxes['scores']):
                np_img = np.round(pred_boxes['masks'][i].cpu().detach().numpy().squeeze())
                rows = np.any(np_img, axis=1)
                cols = np.any(np_img, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                box = {'label': obj_categories[pred_boxes['labels'][i]],
                       'id': obj_categories[pred_boxes['labels'][i]] + f"_{str(i)}",
                       'score': score.cpu().detach().numpy().astype(float) / 1.,
                       'xmin': xmin / width,
                       'ymin': ymin / height,
                       'xmax': xmax / width,
                       'ymax': ymax / height
                       }
                boxes.append(box)

            rel_scores = pred[1]
            pairs = pred[2]

            obj_ids = [x['id'] for x in boxes]

            for i, pair in enumerate(pairs):
                person = obj_ids[pair[0]]
                obj = obj_ids[pair[1]]
                vrb_score = rel_scores[i].cpu().detach().numpy().squeeze()

                rel = {"person_id": person,
                       "obj_id": obj,
                       "score": vrb_score}
                relations.append(rel)
        elif model_info['name'] == "vtranse":
            obj_categories = model_info['classes']
            boxes = []
            relations = []
            pred_boxes = pred[0]
            for i, score in enumerate(pred_boxes['scores']):
                ymin, ymax = pred_boxes['boxes'][i][1], pred_boxes['boxes'][i][3]
                xmin, xmax = pred_boxes['boxes'][i][0], pred_boxes['boxes'][i][2]
                box = {'label': obj_categories[pred_boxes['labels'][i]],
                       'id': obj_categories[pred_boxes['labels'][i]] + f"_{str(i)}",
                       'score': score.cpu().detach().numpy().astype(float) / 1.,
                       'xmin': xmin / width,
                       'ymin': ymin / height,
                       'xmax': xmax / width,
                       'ymax': ymax / height
                       }
                boxes.append(box)

        return cls(boxes=boxes, relations=relations,
                   relations_labels=model_info['relationships'],
                   class_labels=obj_categories,
                   width=width, height=height)


def process_img(img_path, model_info, device):
    model = model_info['model']
    if not isinstance(img_path, Path):
        img_path = Path(img_path)

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    np_img = img.copy()
    og_img = img.copy()
    img = F.to_tensor(img)
    img = img.to(device)
    height, width = np_img.shape[0:2]

    model.eval()
    model.to(device)
    with torch.no_grad():
        out = model([img])

    boxes = BoxDetections.load_model_out(out, width, height, model_info)

    return draw_boxes(np_img, boxes), boxes, og_img


def draw_boxes(img, box_det, threshold=0.3, cmapping='tab20'):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cmap = matplotlib.cm.get_cmap(cmapping)
    thickness = int(np.min([box_det.width/150, box_det.height/150]))

    total_objects = len(box_det[threshold])

    for i, box in enumerate(box_det[threshold]):
        box_mask = np.zeros_like(img)

        top_left = (int(box['xmin']*box_det.width), int(box['ymin']*box_det.height))
        bottom_right = (int(box['xmax']*box_det.width), int(box['ymax']*box_det.height))

        color = np.array(cmap(i / total_objects)) * 255.0
        color = color[::-1]  # convert to bgr

        box_mask = cv2.rectangle(box_mask, top_left, bottom_right, color=color, thickness=thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = box['id']
        box_mask = cv2.putText(box_mask, label, (top_left[0], top_left[1] - 25), font, 1,
                                                (0,255,0), 4, cv2.LINE_AA)
        box_mask = cv2.putText(box_mask, label, (top_left[0], top_left[1] - 25), font, 1,
                                                (255,255,255), 2, cv2.LINE_AA)
        img = cv2.add(img, box_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
