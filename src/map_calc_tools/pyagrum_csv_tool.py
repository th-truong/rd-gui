import pickle
import json
from PIL import Image
from pathlib import Path
from training.data_util.vtranse_dataset import VTranseRelDataset
from tqdm import tqdm
import numpy as np
import pandas as pd


class PyagrumCsvTool():
    def __init__(self, rel_pred_csv_path: Path,
                 metadata_path: Path):
        self.rel_pred_df = pd.read_csv(rel_pred_csv_path)

        pred_labels = self._load_json(metadata_path / "pred_list.json")
        obj_labels = self._load_json(metadata_path / "objects_list.json")

        self.id2predicate = {i: label for i, label in enumerate(pred_labels)}
        self.predicate2id = {label: i for i, label in enumerate(pred_labels)}

        self.id2obj = {i: label for i, label in enumerate(obj_labels)}
        self.obj2id = {label: i for i, label in enumerate(obj_labels)}

        self.binning_functions = {'brisque_score': self.bin_brisque_score,
                                  'sub_box_area': lambda pd_row: self.bin_obj_areas(pd_row, 'sub_box_area'),
                                  'obj_box_area': lambda pd_row: self.bin_obj_areas(pd_row, 'obj_box_area'),
                                  'rel_label': self.bin_pred_relationship_label,
                                  'pred_error': self.bin_pred_error
                                  }

    def save_csv(self, column_labels: list[str], save_path: Path):
        df_to_save = self.rel_pred_df.copy()

        for column_label in column_labels:
            df_to_save[column_label] = df_to_save.apply(self.binning_functions[column_label], axis=1)

        columns_to_drop = [column_name for column_name in df_to_save.columns if column_name not in column_labels]
        df_to_save = df_to_save.drop(columns_to_drop, axis=1)
        df_to_save.to_csv(save_path, index=False)

    @staticmethod
    def bin_brisque_score(pd_row):
        brisque_score = float(pd_row['brisque_score'])
        if 0 <= brisque_score < 10:
            quality = 'Very Good'
        elif 10 <= brisque_score < 20:
            quality = 'Good'
        elif 20 <= brisque_score < 30:
            quality = 'Bad'
        elif 30 <= brisque_score:
            quality = 'Very Bad'
        else:
            quality = np.nan
        return quality

    def bin_pred_relationship_label(self, pd_row):
        pred_rel_label = int(pd_row['pred_label'])
        return self.id2predicate[pred_rel_label]

    @staticmethod
    def bin_obj_areas(pd_row, pd_col: str):
        area = float(pd_row[pd_col])
        if area < 32. ** 2:
            size = 'small'
        elif 32. ** 2 <= area <= 64. ** 2:
            size = 'medium'
        elif 64. ** 2 < area:
            size = 'large'
        return size

    @staticmethod
    def bin_pred_error(pd_row):
        pred_error = int(pd_row['pred_error'])
        if pred_error == 1:
            return 'True'
        elif pred_error == 0:
            return 'False'

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as f:
            contents = json.load(f)
        return contents
