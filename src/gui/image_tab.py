from pathlib import Path

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QLabel, QLineEdit,
                             QListWidget, QPushButton, QTabWidget, QWidget,
                             QTableWidget, QTableView)
from PyQt5.QtCore import QAbstractTableModel, Qt

import pandas as pd
import copy

from training.vrb_model import vrb_full_model


QT_IMG_FORMATS = "All files (*.*);;BMP (*.bmp);;CUR (*.cur);;GIF (*.gif);;ICNS (*.icns);;ICO (*.ico);;JPEG (*.jpeg);;JPG (*.jpg);;PBM (*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);;SVG (*.svg);;SVGZ (*.svgz);;TGA (*.tga);;TIF (*.tif);;TIFF (*.tiff);;WBMP (*.wbmp);;WEBP (*.webp);;XBM (*.xbm);;XPM (*.xpm)"
MODEL_CLASSES = ['background',
                 'backpack',
                 'belt',
                 'dress',
                 'female',
                 'glove',
                 'hat',
                 'jeans',
                 'male',
                 'outerwear',
                 'scarf',
                 'shirt',
                 'shoe',
                 'shorts',
                 'skirt',
                 'sock',
                 'suit',
                 'swim_cap',
                 'swim_wear',
                 'tanktop',
                 'tie',
                 'trousers']

obj_df = pd.DataFrame({'Object': [],
                       'Confidence': []})
relations_df = pd.DataFrame({'Relationship': [],
                             'Confidence': []})


def load_model(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    chkpt_full = torch.load(Path("pretrained_models/optimal_model/optimal_model.tar"))

    model_mrcnn = vrb_full_model.create_mrcnn_model(num_classes=22)
    model_vrb = vrb_full_model.create_full_vrb_model(num_classes=1, model_mrcnn=model_mrcnn)

    model_vrb.load_state_dict(chkpt_full['model'])
    model_vrb.eval()
    model_vrb.to(device)

    return model_vrb, device


class ImageTab(QWidget):
    def __init__(self, parent, cfg):
        super(QWidget, self).__init__(parent)
        self.cfg = cfg
        model_vrb, device = load_model(cfg['model_path'].get())
        self.model = model_vrb
        self.device = device
        layout = QGridLayout(self)
        self.setLayout(layout)

        # textbox for providing path to
        self.pgn_file_txt = QLineEdit(parent)
        self.pgn_file_txt.setText("Select an image to begin.")
        layout.addWidget(self.pgn_file_txt, 0, 0, 1, 7)

        # load image button
        self.open_file_btn = QPushButton("Choose Image")
        self.open_file_btn.clicked.connect(self.open_file_btn_click)
        layout.addWidget(self.open_file_btn, 0, 7, 1, 1)

        # image display
        self.img_widget = QLabel(self)
        layout.addWidget(self.img_widget, 1, 0, 8, 8)
        self.img_widget.setGeometry(10, 10, 1000, 1000)
        self.img_widget.show()

        # tables for displaying results
        self.results_tables = ResultsTab(self)
        # self.engine_list.clicked.connect(self.engine_list_click)
        layout.addWidget(self.results_tables, 0, 8, 4, 7)

    def open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            self.cfg['imgs_folder'].get(),
                                            QT_IMG_FORMATS)
        if fname[0] != "":
            fname = Path(str(fname[0]))
            self.pgn_file_txt.setText(str(fname))

            if Path(self.pgn_file_txt.text()).exists():
                img, boxes, og_img = process_img(fname, self.model, self.device)
                self.og_img = og_img
                self.results_tables.boxes = boxes
                self.update_display_img(img)
                self.update_detections_tables(boxes)

    def update_display_img(self, img):
        Qimg_obj = QImage(img, img.shape[1], img.shape[0],
                          QImage.Format_RGB888)
        self.img_widget.setPixmap(QPixmap(Qimg_obj).scaled(1280, 720, aspectRatioMode=1))

    def update_detections_tables(self, boxes):
        self.results_tables.objects_table.update_data(boxes.boxes_as_df())
        self.results_tables.relations_table.update_data(boxes.relations_as_df())


class ResultsTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.parent = parent

        self.boxes = []

        self.objects_table = PandasQTableModel(obj_df)
        self.objects_view = QTableView()
        self.objects_view.setModel(self.objects_table)
        self.objects_view.clicked.connect(self.object_clicked)

        self.relations_table = PandasQTableModel(relations_df)
        self.relations_view = QTableView()
        self.relations_view.setModel(self.relations_table)
        self.relations_view.clicked.connect(self.relation_clicked)

        self.tabs.addTab(self.objects_view, "Objects")
        self.tabs.addTab(self.relations_view, "Relations")

        layout = QGridLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def relation_clicked(self, item):
        idx = item.row()
        chosen_row = self.boxes.relations[idx]
        person_box = self.boxes[chosen_row['person_id']]
        obj_box = self.boxes[chosen_row['obj_id']]
        boxes = person_box + obj_box
        # TODO: do this better, don't make a deep copy...
        new_boxes = copy.deepcopy(self.boxes)
        new_boxes.detections = boxes

        img = self.parent.og_img
        self.parent.update_display_img(draw_boxes(img, new_boxes, threshold=0.))

    def object_clicked(self, item):
        idx = item.row()
        chosen_row = self.boxes[int(idx)]
        # TODO: do this better, don't make a deep copy...
        new_boxes = copy.deepcopy(self.boxes)
        new_boxes.detections = chosen_row

        img = self.parent.og_img
        self.parent.update_display_img(draw_boxes(img, new_boxes, threshold=0.))


class PandasQTableModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

    def update_data(self, data):
        self._data = data
        self.layoutChanged.emit()


from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as F
import cv2
import matplotlib.cm
# TODO: clean this up and move it somewhere else

class BoxDetections():
    def __init__(self, boxes, relations, width=None, height=None):
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

    def boxes_as_df(self):
        scores = [x['score'] for x in self.detections]
        labels = [x['id'] for x in self.detections]
        return pd.DataFrame({'Object': labels,
                             'Confidence': scores})

    def relations_as_df(self):
        scores = [x['score'] for x in self.relations]
        relations = [x['person_id'] + " wears " + x['obj_id'] for x in self.relations]
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
            if key in MODEL_CLASSES:
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
    def load_model_out(cls, pred, width, height,
                       obj_categories=MODEL_CLASSES):
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

        return cls(boxes=boxes, relations=relations, width=width, height=height)

def process_img(img_path, model, device):
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
    labels = out[0]['labels']
    scores = out[0]['scores']
    masks = out[0]['masks']

    boxes = BoxDetections.load_model_out(out, width, height)

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
                                                (255,255,255), 2, cv2.LINE_AA)
        img = cv2.add(img, box_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
