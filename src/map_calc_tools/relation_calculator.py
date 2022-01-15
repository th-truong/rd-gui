import pickle
import json
from PIL import Image
from pathlib import Path
from training.data_util.vtranse_dataset import VTranseRelDataset
from tqdm import tqdm
import numpy as np
import pandas as pd


class RelationshipCalculator():
    def __init__(self,
                 vtranse_ds: VTranseRelDataset,
                 brisque_path: Path,
                 rel_pred_path: Path,
                 min_max_brisque: tuple[int, int] = (0, 100)):
        with open(brisque_path, "r") as f:
            self.brisque_scores = json.load(f)

        with open(rel_pred_path, 'rb') as f:
            self.rel_predictions = pickle.load(f)

        self.min_max_brisque = min_max_brisque
        self.vtranse_ds = vtranse_ds

    @property
    def filtered_brisque_scores(self):
        min_score = self.min_max_brisque[0]
        max_score = self.min_max_brisque[1]

        filtered_brisque_scores = {image_id: score for image_id, score in self.brisque_scores.items()
                                   if min_score <= score <= max_score}
        return filtered_brisque_scores

    def calc_tp_fp_fn(self,
                      score_thresh=0.2):
        class_metrics = {key: {'tp': 0,
                               'fp': 0,
                               'fn': 0}
                         for key in self.vtranse_ds.id2predicate}

        for image_id in tqdm(self.filtered_brisque_scores, total=len(self.filtered_brisque_scores)):
            prediction = self.rel_predictions[image_id]

            if prediction['rel_labels'].size == 1:
                label_score_iterator = zip([prediction['rel_labels'].tolist()], prediction['scores'])
            else:
                label_score_iterator = zip(prediction['rel_labels'], prediction['scores'])

            for label, scores in label_score_iterator:
                rel_predictions = np.where(scores > score_thresh)[0]
                if label in rel_predictions:
                    class_metrics[label]['tp'] += 1
                    rel_predictions = np.delete(rel_predictions, np.where(rel_predictions == label))
                else:
                    class_metrics[label]['fn'] += 1

                for rel_prediction in rel_predictions:
                    class_metrics[rel_prediction]['fp'] += 1
        return class_metrics

    def to_pyagrum_csv(self, save_path, score_thresh=0.5):
        # columns
        pyagrum_dict = {'image_id': [],
                        'sub_box_area': [],
                        'obj_box_area': [],
                        'sub_label': [],
                        'obj_label': [],
                        'pred_label': [],
                        'img_area': [],
                        'brisque_score': [],
                        'pred_error': []}

        for image_id, brisque_score in tqdm(self.filtered_brisque_scores.items(), total=len(self.filtered_brisque_scores)):
            prediction = self.rel_predictions[image_id]
            if prediction['rel_labels'].size == 1:
                label_score_iterator = zip([prediction['rel_labels'].tolist()],
                                           prediction['scores'],
                                           prediction['sub_boxes'],
                                           prediction['obj_boxes'],
                                           prediction['sub_labels'],
                                           prediction['obj_labels']
                                           )
            else:
                label_score_iterator = zip(prediction['rel_labels'],
                                           prediction['scores'],
                                           prediction['sub_boxes'],
                                           prediction['obj_boxes'],
                                           prediction['sub_labels'],
                                           prediction['obj_labels']
                                           )

            for label, scores, sub_box, obj_box, sub_label, obj_label in label_score_iterator:
                rel_predictions = np.where(scores > score_thresh)[0]
                if label in rel_predictions:
                    pred_error = 0
                    pred_label = label
                else:
                    pred_error = 1
                    pred_label = np.argmax(scores)

                pyagrum_dict['image_id'].append(image_id)
                pyagrum_dict['sub_box_area'].append(self._get_obj_area(*sub_box))
                pyagrum_dict['obj_box_area'].append(self._get_obj_area(*obj_box))
                pyagrum_dict['sub_label'].append(sub_label)
                pyagrum_dict['obj_label'].append(obj_label)
                pyagrum_dict['pred_label'].append(pred_label)
                pyagrum_dict['img_area'].append(prediction['width'] * prediction['height'])
                pyagrum_dict['brisque_score'].append(brisque_score)
                pyagrum_dict['pred_error'].append(pred_error)

        pyagrum_pd = pd.DataFrame.from_dict(pyagrum_dict)
        pyagrum_pd.to_csv(save_path, index=False)

    @staticmethod
    def _get_obj_area(xmin, ymin, xmax, ymax):
        return float((xmax - xmin) * (ymax - ymin))

    @staticmethod
    def calc_recall(class_metrics,
                    mode='micro'):
        recall = {key: None for key in class_metrics}

        if mode == 'macro':
            for class_number, metrics in class_metrics.items():
                if metrics['tp'] + metrics['fn'] > 0:
                    recall[class_number] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
            return np.mean([recall_value for recall_value in recall.values() if recall_value is not None])
        elif mode == 'micro':
            tp = np.sum([metrics['tp'] for metrics in class_metrics.values()])
            fn = np.sum([metrics['fn'] for metrics in class_metrics.values()])
            return tp / (tp + fn)
        elif mode == 'individual':
            for class_number, metrics in class_metrics.items():
                if metrics['tp'] + metrics['fn'] > 0:
                    recall[class_number] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
            return recall

    @staticmethod
    def calc_precision(class_metrics,
                       mode='micro'):
        precision = {key: None for key in class_metrics}

        if mode == 'macro':
            for class_number, metrics in class_metrics.items():
                if metrics['tp'] + metrics['fp'] > 0:
                    precision[class_number] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
            return np.mean([precision_value for precision_value in precision.values() if precision_value is not None])
        elif mode == 'micro':
            tp = np.sum([metrics['tp'] for metrics in class_metrics.values()])
            fp = np.sum([metrics['fp'] for metrics in class_metrics.values()])
            return tp / (tp + fp)
        elif mode == 'individual':
            for class_number, metrics in class_metrics.items():
                if metrics['tp'] + metrics['fp'] > 0:
                    precision[class_number] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
            return precision
