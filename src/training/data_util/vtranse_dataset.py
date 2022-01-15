from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import Dataset
import json
import matplotlib.cm
from torchvision.transforms import functional as F
import torch
import pickle


class VTranseDataset(Dataset):
    def __init__(self, dataset_path: Path, ds_set="train"):
        super(VTranseDataset).__init__()
        metadata_path = dataset_path / "metadata"
        self.ds_set = ds_set
        self._prepare_labels(metadata_path)

        ds_set_path = dataset_path /  ("vtranse_" + ds_set + "_labels.pkl")
        with open(ds_set_path, "rb") as f:
            self.samples = pickle.load(f)
        self.dataset_path = dataset_path

    def view_sample(self, idx, rel_idx):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        fig, ax = plt.subplots()
        ax.imshow(img)

        colours = ['red', 'cyan']

        subject_boxes = np.array(sample['sub_boxes'])
        object_boxes = np.array(sample['obj_boxes'])
        labels = np.array(sample['rlp_labels'])

        x = [subject_boxes[rel_idx, 0], object_boxes[rel_idx, 0]]
        y = [subject_boxes[rel_idx, 1], object_boxes[rel_idx, 1]]
        box_width = [subject_boxes[rel_idx, 2] - x[0], object_boxes[rel_idx, 2] - x[1]]
        box_height = [subject_boxes[rel_idx, 3] - y[0], object_boxes[rel_idx, 3] - y[1]]

        names = [self.id2obj[labels[rel_idx, 0]],
                 self.id2obj[labels[rel_idx, 2]]]

        rect = []
        for i in range(len(x)):
            rect.append(mpatches.Rectangle((x[i], y[i]), box_width[i], box_height[i],
                                           fill=False, linewidth=2, edgecolor=colours[i]))
            ax.add_patch(rect[i])
            plt.text(x[i], y[i], names[i],
                     bbox=dict(color=colours[i])
                     )
        title = [names[0], self.id2predicate[labels[rel_idx, 1]], names[1]]
        title = [str(x) for x in title]
        title = " ".join(title)
        plt.title(title)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.ds_set == "train":
            sample = self.rels[self.rels.image_id == int(self.train_ds[idx])].iloc[0]
        elif self.ds_set == "test":
            sample = self.rels[self.rels.image_id == int(self.test_ds[idx])].iloc[0]
        for relationship in sample:
            print(relationship)
        return sample

    def _load_json(self, path):
        with open(path, 'r') as f:
            contents = json.load(f)
        return contents

    def _prepare_labels(self, metadata_path):
        if self.ds_set == "train":
            self.ds_idxs = self._load_json(metadata_path / "train_list.json")
        else:
            self.ds_idxs = self._load_json(metadata_path / "test_list.json")

        pred_labels = self._load_json(metadata_path / "pred_list.json")
        obj_labels = self._load_json(metadata_path / "objects_list.json")

        self.predicate_set = set(pred_labels)
        self.obj_set = set(obj_labels)

        self.id2predicate = {i: label for i, label in enumerate(pred_labels)}
        self.predicate2id = {label: i for i, label in enumerate(pred_labels)}

        self.id2obj = {i: label for i, label in enumerate(obj_labels)}
        self.obj2id = {label: i for i, label in enumerate(obj_labels)}


class VTranseObjDataset(VTranseDataset):
    def __init__(self, dataset_path: Path, ds_set="train"):
        super(VTranseObjDataset, self).__init__(dataset_path, ds_set=ds_set)

    def view_sample(self, idx, cmapping='viridis', max_boxes=5):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        boxes_labels = self._get_unique_boxes(sample)

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        fig, ax = plt.subplots()
        ax.imshow(img)
        cmap = matplotlib.cm.get_cmap(cmapping)

        num_boxes = boxes_labels.shape[0]

        for i, box_label in enumerate(boxes_labels):
            x = box_label[0]
            y = box_label[1]
            width = box_label[2] - x
            height = box_label[3] - y
            rect = mpatches.Rectangle((x, y), width, height,
                                      fill=False, linewidth=2, edgecolor=cmap(i/num_boxes))
            ax.add_patch(rect)
            name = self.id2obj[box_label[4]]
            plt.text(x, y, name,
                     bbox=dict(color=cmap(i/num_boxes)))
            if i >= max_boxes:
                break
        title = f"Bounding Boxes for image {image_id}.jpg"
        plt.title(title)

    def __getitem__(self, idx):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        boxes_labels = self._get_unique_boxes(sample)

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        labels = boxes_labels[:, -1]
        boxes = boxes_labels[:, 0:-1]

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes,
                  'labels': labels}

        return img, target

    def _get_unique_boxes(self, sample):

        sub_boxes = np.array(sample['sub_boxes'])
        obj_boxes = np.array(sample['obj_boxes'])

        rlp_labels = np.array(sample['rlp_labels'])

        boxes_labels = np.concatenate((rlp_labels[:, 0],
                                       rlp_labels[:, 2]))

        boxes = np.concatenate((sub_boxes, obj_boxes))

        uniq_boxes_labels = np.unique(np.hstack((boxes, boxes_labels[:, None])),
                                      axis=0)

        return uniq_boxes_labels


class VTranseRelDataset(VTranseObjDataset):
    # this one loads an image and relationships without processing multi labels for the same sub-obj pair
    # targets are not one hot encoded
    def __init__(self, dataset_path: Path, ds_set="train"):
        super(VTranseRelDataset, self).__init__(dataset_path, ds_set=ds_set)

    def view_sample(self, idx, rel_idx=0, cmapping='viridis'):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        sub_box = np.array(sample['sub_boxes'][rel_idx])
        obj_box = np.array(sample['obj_boxes'][rel_idx])
        boxes_labels = [sub_box, obj_box]

        sub_label = self.id2obj[sample['rlp_labels'][rel_idx][0]]
        obj_label = self.id2obj[sample['rlp_labels'][rel_idx][2]]
        labels = [sub_label, obj_label]
        rel_label = self.id2predicate[sample['rlp_labels'][rel_idx][1]]

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        fig, ax = plt.subplots()
        ax.imshow(img)
        cmap = matplotlib.cm.get_cmap(cmapping)

        num_boxes = len(boxes_labels)

        for i, (box_label, name) in enumerate(zip(boxes_labels, labels)):
            x = box_label[0]
            y = box_label[1]
            width = box_label[2] - x
            height = box_label[3] - y
            rect = mpatches.Rectangle((x, y), width, height,
                                      fill=False, linewidth=2, edgecolor=cmap(i/num_boxes))
            ax.add_patch(rect)
            plt.text(x, y, name,
                     bbox=dict(color=cmap(i/num_boxes)))
        title = f"Bounding Boxes for image {image_id}.jpg. '{rel_label}'"
        plt.title(title)

    def __getitem__(self, idx):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        boxes_labels = self._get_unique_boxes(sample)

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        labels = boxes_labels[:, -1]
        boxes = boxes_labels[:, 0:-1]

        rels = zip(sample['sub_boxes'], sample['rlp_labels'], sample['obj_boxes'])
        rel_labels = []
        obj_box_masks = []
        sub_box_masks = []
        obj_boxes = []
        obj_labels = []
        sub_boxes = []
        sub_labels = []
        for sub_box, rel, obj_box in rels:
            rel_labels.append(rel[1])
            sub_labels.append(rel[0])
            obj_labels.append(rel[2])
            sub_boxes.append(sub_box)
            obj_boxes.append(obj_box)

            obj_box_mask = self._create_box_image(obj_box, img.shape)
            obj_box_mask = F.to_tensor(obj_box_mask)
            obj_box_masks.append(obj_box_mask)

            sub_box_mask = self._create_box_image(sub_box, img.shape)
            sub_box_mask = F.to_tensor(sub_box_mask)
            sub_box_masks.append(sub_box_mask)

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        rel_labels = torch.as_tensor(rel_labels, dtype=torch.int64)

        target = {'boxes': boxes,
                  'labels': labels,
                  'rel_labels': rel_labels}

        return img, target, sub_box_masks, obj_box_masks, sub_boxes, obj_boxes, sub_labels, obj_labels

    def _create_box_image(self, box_coords, img_shape) -> np.ndarray:
        mask = np.zeros(img_shape[0:2], dtype=np.uint8)
        mask[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]] = 255
        return mask


class VTranseRelTrainDataset(VTranseRelDataset):
    # this one loads an image and relationships and processes any multi labels for sub-obj pairs
    # targets are one hot encoded now because of multi label
    def __init__(self, dataset_path: Path, ds_set="train"):
        super(VTranseRelTrainDataset, self).__init__(dataset_path, ds_set=ds_set)

    def __getitem__(self, idx):
        image_id = self.ds_idxs[idx]
        sample = self.samples[image_id]

        boxes_labels = self._get_unique_boxes(sample)

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (image_id + ".jpg")))
        labels = boxes_labels[:, -1]
        boxes = boxes_labels[:, 0:-1]

        rels = list(zip(sample['sub_boxes'], sample['rlp_labels'], sample['obj_boxes']))

        # need to convert lists to tuples to be able to catch duplicates using set b/c lists arent hashable
        sub_boxes = [tuple(box) for box in sample['sub_boxes']]
        obj_boxes = [tuple(box) for box in sample['obj_boxes']]
        sub_obj_pairs = list(zip(sub_boxes, obj_boxes))
        duplicates = list([(ele, i) for i, ele in enumerate(sub_obj_pairs)
                          if sub_obj_pairs.count(ele) > 1])
        duplicate_indices = [x[1] for x in duplicates]
        duplicate_sub_obj_pairs = set([x[0] for x in duplicates])

        duplicate_rels = [rels[i] for i in duplicate_indices]
        rels = [x for x in rels if x not in duplicate_rels]
        rel_labels = []
        obj_box_masks = []
        sub_box_masks = []

        # make all boxes and labels for non-duplicates
        for sub_box, rel, obj_box in rels:
            rel_label = np.zeros(100)  # 100 relationship classes
            rel_label[rel[1]] = 1.

            rel_labels.append(rel_label)

            obj_box_mask = self._create_box_image(obj_box, img.shape)
            obj_box_mask = F.to_tensor(obj_box_mask)
            obj_box_masks.append(obj_box_mask)

            sub_box_mask = self._create_box_image(sub_box, img.shape)
            sub_box_mask = F.to_tensor(sub_box_mask)
            sub_box_masks.append(sub_box_mask)

        # work through duplicates
        for sub_box, obj_box in duplicate_sub_obj_pairs:
            sub_box = list(sub_box)
            obj_box = list(obj_box)

            obj_box_mask = self._create_box_image(obj_box, img.shape)
            obj_box_mask = F.to_tensor(obj_box_mask)
            obj_box_masks.append(obj_box_mask)

            sub_box_mask = self._create_box_image(sub_box, img.shape)
            sub_box_mask = F.to_tensor(sub_box_mask)
            sub_box_masks.append(sub_box_mask)

            rel_label = np.zeros(100)  # 100 relationship classes

            for sub_box_dupe, rel, obj_box_dupe in duplicate_rels:
                if sub_box_dupe == sub_box and obj_box_dupe == obj_box:
                    rel_label[rel[1]] = 1.
            # some of the duplicates are duplicated in label too so won't always sum to greater than 1
            rel_labels.append(rel_label)

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        rel_labels = torch.as_tensor(rel_labels, dtype=torch.float32)

        target = {'boxes': boxes,
                  'labels': labels,
                  'rel_labels': rel_labels}

        return img, target, sub_box_masks, obj_box_masks
