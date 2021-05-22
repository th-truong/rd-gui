from pathlib import Path

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QLabel, QLineEdit,
                             QListWidget, QPushButton, QTabWidget, QWidget,
                             QTableWidget, QTableView)
from PyQt5.QtCore import QAbstractTableModel, Qt

import pandas as pd
from scripts import load_model


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
obj_df = pd.DataFrame({'Object': ['man_0', 'chair_0', 'apple_0'],
                   'Confidence': [0.99, 0.54, 0.42]})
relations_df = pd.DataFrame({'Relationship': ['man_0 holds apple_0', 'man_0 sits on chair_0',
                                              'man_0 eats apple_0'],
                   'Confidence': [0.99, 0.54, 0.42]})


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
        fname = Path(str(fname[0]))
        self.pgn_file_txt.setText(str(fname))

        if Path(self.pgn_file_txt.text()).exists():
            img = process_img(fname, self.model, self.device)
            Qimg_obj = QImage(img, img.shape[1], img.shape[0],
                              QImage.Format_RGB888)
            self.img_widget.setPixmap(QPixmap(Qimg_obj).scaled(1280, 720, aspectRatioMode=1))


class ResultsTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()

        self.objects_table = PandasQTableModel(obj_df)
        self.objects_view = QTableView()
        self.objects_view.setModel(self.objects_table)

        self.relations_table = PandasQTableModel(relations_df)
        self.relations_view = QTableView()
        self.relations_view.setModel(self.relations_table)

        self.tabs.addTab(self.objects_view, "Objects")
        self.tabs.addTab(self.relations_view, "Relations")

        layout = QGridLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)


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
    def __init__(self, boxes, width=None, height=None):
        """[summary]

        Args:
            boxes ([type]): [description]
            width ([type], optional): [description]. Defaults to None.
            height ([type], optional): [description]. Defaults to None.
        """
        self.detections = boxes
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
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
        pred = pred[0]
        for i, score in enumerate(pred['scores']):
            np_img = np.round(pred['masks'][i].cpu().detach().numpy().squeeze())
            rows = np.any(np_img, axis=1)
            cols = np.any(np_img, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            box = {'label': MODEL_CLASSES[pred['labels'][i]],
                   'id': MODEL_CLASSES[pred['labels'][i]] + f"_{str(i)}",
                   'score': score.cpu().detach().numpy().astype(float) / 1.,
                   'xmin': xmin / width,
                   'ymin': ymin / height,
                   'xmax': xmax / width,
                   'ymax': ymax / height
                   }
            boxes.append(box)
        return cls(boxes=boxes, width=width, height=height)


def process_img(img_path, model, device):
    if not isinstance(img_path, Path):
        img_path = Path(img_path)

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    np_img = img.copy()
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

    return draw_boxes(np_img, boxes)


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
