import pickle
import json
from PIL import Image
from pathlib import Path
from training.data_util.vtranse_dataset import VTranseObjDataset


class CocoWriter():
    def __init__(self,
                 brisque_path: Path,
                 obj_path: Path,
                 images_path: Path,
                 gt_dataset: VTranseObjDataset,
                 min_max_brisque: tuple[int, int] = (0, 100)):
        with open(brisque_path, "r") as f:
            self.brisque_scores = json.load(f)

        with open(obj_path, "rb") as f:
            self.obj_pred = pickle.load(f)

        self.images_path = images_path
        self.min_max_brisque = min_max_brisque

        self.gt_dataset = gt_dataset
        self.img_id_2_ds_idx = {image_id: i for i, image_id in enumerate(gt_dataset.ds_idxs)}

    @property
    def filtered_brisque_scores(self):
        min_score = self.min_max_brisque[0]
        max_score = self.min_max_brisque[1]

        filtered_brisque_scores = {image_id: score for image_id, score in self.brisque_scores.items()
                                   if min_score <= score <= max_score}
        return filtered_brisque_scores

    def save_gt_coco_json(self, save_path: Path):
        images = []
        annotations = []
        num_objects = 0
        for image_id in self.filtered_brisque_scores.keys():
            file_name = image_id + ".jpg"
            width, height = self._get_image_size(image_id)
            image = {"id": image_id,
                     "file_name": file_name,
                     "width": width,
                     "height": height}

            img_targets = self.gt_dataset[self.img_id_2_ds_idx[image_id]][1]
            bboxes = img_targets['boxes'].tolist()
            labels = img_targets['labels'].tolist()

            for i, label in enumerate(labels):
                id = num_objects + i
                bbox = bboxes[i]
                area = self._get_obj_area(*bbox)

                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                coco_bbox = [bbox[0], bbox[1], bbox_width, bbox_height]

                annotation = {'id': id,
                              'segmentation': [],
                              'area': area,
                              'iscrowd': 0,
                              'ignore': 0,
                              'image_id': image_id,
                              'bbox': coco_bbox,
                              'category_id': label}
                annotations.append(annotation)
            num_objects += len(labels)

            images.append(image)

        categories = [{'id': class_id,
                       'name': name,
                       'supercategory': "none"} for class_id, name in self.gt_dataset.id2obj.items()]
        categories = categories[1:]  # remove the first id, 0, because it is background
        coco_dict = {"images": images,
                     "annotations": annotations,
                     "categories": categories}
        with open(save_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

    def save_det_coco_json(self, save_path: Path,
                           score_thresh=0.5):
        images = []
        annotations = []
        num_objects = 0
        for image_id in self.filtered_brisque_scores.keys():
            file_name = image_id + ".jpg"
            width, height = self._get_image_size(image_id)
            image = {"id": image_id,
                     "file_name": file_name,
                     "width": width,
                     "height": height}

            img_targets = self.obj_pred[image_id]
            bboxes = [box.tolist() for i, box in enumerate(list(img_targets['boxes']))
                      if img_targets['scores'][i] >= score_thresh]
            labels = [label.tolist() for i, label in enumerate(img_targets['labels'])
                      if img_targets['scores'][i] >= score_thresh]
            scores = [float(score) for score in img_targets['scores']
                      if score >= score_thresh]

            for i, label in enumerate(labels):
                id = num_objects + i
                bbox = bboxes[i]
                area = self._get_obj_area(*bbox)

                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                coco_bbox = [bbox[0], bbox[1], bbox_width, bbox_height]

                annotation = {'id': id,
                              'segmentation': [],
                              'area': area,
                              'iscrowd': 0,
                              'ignore': 0,
                              'image_id': image_id,
                              'bbox': coco_bbox,
                              'category_id': label,
                              'score': scores[i]}
                annotations.append(annotation)
            num_objects += len(labels)

            images.append(image)

        categories = [{'id': class_id,
                       'name': name,
                       'supercategory': "none"} for class_id, name in self.gt_dataset.id2obj.items()]
        categories = categories[1:]  # remove the first id, 0, because it is background
        coco_dict = {"images": images,
                     "annotations": annotations,
                     "categories": categories}
        with open(save_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

    def _get_image_size(self, image_id):
        img = Image.open(self.images_path / (image_id + ".jpg"))
        width, height = img.size
        return width, height

    @staticmethod
    def _get_obj_area(xmin, ymin, xmax, ymax):
        return float((xmax - xmin) * (ymax - ymin))
