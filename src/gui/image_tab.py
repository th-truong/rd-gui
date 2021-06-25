from pathlib import Path

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QLabel, QLineEdit,
                             QListWidget, QPushButton, QTabWidget, QWidget,
                             QTableWidget, QTableView)
from PyQt5.QtCore import QAbstractTableModel, Qt

import pandas as pd
import copy
import torch
import json
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from training.vrb_model import vrb_full_model
from gui.model_dataclasses import bounding_boxes


QT_IMG_FORMATS = "All files (*.*);;BMP (*.bmp);;CUR (*.cur);;GIF (*.gif);;ICNS (*.icns);;ICO (*.ico);;JPEG (*.jpeg);;JPG (*.jpg);;PBM (*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);;SVG (*.svg);;SVGZ (*.svgz);;TGA (*.tga);;TIF (*.tif);;TIFF (*.tiff);;WBMP (*.wbmp);;WEBP (*.webp);;XBM (*.xbm);;XPM (*.xpm)"


def load_model(models_info, name):
    model_path = models_info[name]['model_path']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if name == "relatable_clothing":
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        chkpt_full = torch.load(Path("pretrained_models/optimal_model/optimal_model.tar"))

        model_mrcnn = vrb_full_model.create_mrcnn_model(num_classes=22)
        model = vrb_full_model.create_full_vrb_model(num_classes=1, model_mrcnn=model_mrcnn)

        model.load_state_dict(chkpt_full['model'])
        model.eval()
        model.to(device)
    elif name == "vtranse":
        model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=201,
                                        trainable_backbone_layers=5)
        model_dict = torch.load(models_info[name]['model_path'])
        model.load_state_dict(model_dict['model'])
        model.eval()
        model.to(device)

    return model


def load_label_json(json_path):
    with open(json_path, 'r') as f:
        labels = json.load(f)
    return labels


class ImageTab(QWidget):
    def __init__(self, parent, cfg):
        super(QWidget, self).__init__(parent)
        self.cfg = cfg

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        models_info = cfg['models'].get()
        self.models = [{'name': name,
                        'model': load_model(models_info, name),
                        'classes': load_label_json(models_info[name]['classes']),
                        'relationships': load_label_json(models_info[name]['relationships'])
                        } for name in models_info.keys()]
        self.model = self.models[0]

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

        # listbox for models
        self.model_listbox = QListWidget()
        self.model_listbox.clicked.connect(self.model_list_click)
        layout.addWidget(self.model_listbox, 4, 8, 4, 7)
        for i, model in enumerate(self.models):
            self.model_listbox.insertItem(i, model['name'])

    def open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            self.cfg['imgs_folder'].get(),
                                            QT_IMG_FORMATS)
        if fname[0] != "":
            fname = Path(str(fname[0]))
            self.pgn_file_txt.setText(str(fname))

            if Path(self.pgn_file_txt.text()).exists():
                img, boxes, og_img = bounding_boxes.process_img(fname, self.model, self.device)
                self.og_img = og_img
                self.results_tables.boxes = boxes
                self.update_display_img(img)
                self.update_detections_tables(boxes)

    def model_list_click(self):
        # set game as current selected game from list and display it
        self.model = self.models[self.model_listbox.currentRow()]

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

        obj_df = pd.DataFrame({'Object': [],
                               'Confidence': []})
        relations_df = pd.DataFrame({'Relationship': [],
                                     'Confidence': []})

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
        self.parent.update_display_img(bounding_boxes.draw_boxes(img, new_boxes, threshold=0.))

    def object_clicked(self, item):
        idx = item.row()
        chosen_row = self.boxes[int(idx)]
        # TODO: do this better, don't make a deep copy...
        new_boxes = copy.deepcopy(self.boxes)
        new_boxes.detections = [chosen_row]

        img = self.parent.og_img
        self.parent.update_display_img(bounding_boxes.draw_boxes(img, new_boxes, threshold=0.))


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
