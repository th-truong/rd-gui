from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import Dataset
import h5py
import json


class VGRelationsDataset(Dataset):
    def __init__(self, dataset_path: Path, set="train"):
        super(VGRelationsDataset).__init__()
        self.rels = pd.read_pickle(dataset_path / "relationships.pkl")

        metadata_path = dataset_path / "metadata"

        self._prepare_labels(metadata_path)

        self.set = set

        self.dataset_path = dataset_path

    def view_sample(self, idx, rel_idx):
        if self.set == "train":
            sample = self.rels[self.rels.image_id == int(self.train_ds[idx])].iloc[0]
        elif self.set == "test":
            sample = self.rels[self.rels.image_id == int(self.test_ds[idx])].iloc[0]

        if rel_idx < len(sample.relationships):
            rel = sample.relationships[rel_idx]
        else:
            rel = sample.relationships[-1]

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (str(sample.image_id) + ".jpg")))
        fig, ax = plt.subplots()
        ax.imshow(img)

        colours = ['red', 'cyan']

        x = [rel['subject']['x'], rel['object']['x']]
        y = [rel['subject']['y'], rel['object']['y']]
        box_width = [rel['subject']['w'], rel['object']['w']]
        box_height = [rel['subject']['h'], rel['object']['h']]

        names = []
        names.append(_get_rel_name(rel['subject']))
        names.append(_get_rel_name(rel['object']))

        rect = []
        for i in range(len(x)):
            rect.append(mpatches.Rectangle((x[i], y[i]), box_width[i], box_height[i],
                                           fill=False, linewidth=2, edgecolor=colours[i]))
            ax.add_patch(rect[i])
            plt.text(x[i], y[i], names[i],
                     bbox=dict(color=colours[i])
                     )
        title = [names[0], rel['predicate'], names[1]]
        title = " ".join(title)
        plt.title(title)

    def __len__(self):
        if set == "train":
            return len(self.train_ds)
        elif set == "test":
            return len(self.test_ds)

    def __getitem__(self, idx):
        if self.set == "train":
            sample = self.rels[self.rels.image_id == int(self.train_ds[idx])].iloc[0]
        elif self.set == "test":
            sample = self.rels[self.rels.image_id == int(self.test_ds[idx])].iloc[0]
        for relationship in sample:
            print(relationship)
        return sample

    def _load_json(self, path):
        with open(path, 'r') as f:
            contents = json.load(f)
        return contents

    def _prepare_labels(self, metadata_path):
        self.test_ds = self._load_json(metadata_path / "test_list.json")
        self.train_ds = self._load_json(metadata_path / "train_list.json")

        pred_labels = self._load_json(metadata_path / "pred_list.json")
        obj_labels = self._load_json(metadata_path / "objects_list.json")

        self.predicate_set = set(pred_labels)
        self.obj_set = set(obj_labels)

        self.id2predicate = {i: label for i, label in enumerate(pred_labels)}
        self.predicate2id = {label: i for i, label in enumerate(pred_labels)}

        self.id2obj = {i: label for i, label in enumerate(obj_labels)}
        self.obj2id = {label: i for i, label in enumerate(obj_labels)}


class VGAttributesDataset(Dataset):
    def __init__(self, dataset_path: Path, set="train"):
        super(VGAttributesDataset).__init__()
        self.relationships = pd.read_json(dataset_path / "relationships.json")
        self.image_data = pd.read_json(dataset_path / "image_data.json")
        self.attributes = pd.read_json(dataset_path / "attributes.json")

        ds_splits = h5py.File(dataset_path / "vg1_2_meta.h5", 'r')
        self.test_ds = list(ds_splits['gt']['test'])
        self.train_ds = list(ds_splits['gt']['train'])

        self.set = set

        self.dataset_path = dataset_path

    def view_sample(self, idx):
        pass

    def __getitem__(self, idx):
        pass


class VGObjectsDataset(Dataset):
    def __init__(self, dataset_path: Path, set="train"):
        super(VGObjectsDataset).__init__()
        self.objs = pd.read_pickle(dataset_path / "objects.pkl")

        ds_splits = h5py.File(dataset_path / "vg1_2_meta.h5", 'r')
        self.test_ds = list(ds_splits['gt']['test'])
        self.train_ds = list(ds_splits['gt']['train'])

        self.set = set

        self.dataset_path = dataset_path

    def view_sample(self, idx, rel_idx):
        if self.set == "train":
            sample = self.rels[self.rels.image_id == int(self.train_ds[idx])].iloc[0]
        elif self.set == "test":
            sample = self.rels[self.rels.image_id == int(self.test_ds[idx])].iloc[0]

        if rel_idx < len(sample.relationships):
            rel = sample.relationships[rel_idx]
        else:
            rel = sample.relationships[-1]

        img = np.array(Image.open(self.dataset_path / "VG_100K" / (str(sample.image_id) + ".jpg")))
        fig, ax = plt.subplots()
        ax.imshow(img)

        colours = ['red', 'cyan']

        x = [rel['subject']['x'], rel['object']['x']]
        y = [rel['subject']['y'], rel['object']['y']]
        box_width = [rel['subject']['w'], rel['object']['w']]
        box_height = [rel['subject']['h'], rel['object']['h']]

        names = []
        names.append(_get_rel_name(rel['subject']))
        names.append(_get_rel_name(rel['object']))

        rect = []
        for i in range(len(x)):
            rect.append(mpatches.Rectangle((x[i], y[i]), box_width[i], box_height[i],
                                           fill=False, linewidth=2, edgecolor=colours[i]))
            ax.add_patch(rect[i])
            plt.text(x[i], y[i], names[i],
                     bbox=dict(color=colours[i])
                     )
        title = [names[0], rel['predicate'], names[1]]
        title = " ".join(title)
        plt.title(title)

    def __len__(self):
        if set == "train":
            return len(self.train_ds)
        elif set == "test":
            return len(self.test_ds)

    def __getitem__(self, idx):
        if self.set == "train":
            sample = self.rels[self.rels.image_id == int(self.train_ds[idx])].iloc[0]
        elif self.set == "test":
            sample = self.rels[self.rels.image_id == int(self.test_ds[idx])].iloc[0]
        for relationship in sample:
            print(relationship)
        return sample


def _get_rel_name(subj_or_obj):
    if "name" in list(subj_or_obj.keys()):
        return subj_or_obj['name']
    else:
        return subj_or_obj['names'][0]
