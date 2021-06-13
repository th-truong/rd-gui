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
