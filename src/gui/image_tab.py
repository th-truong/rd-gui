from pathlib import Path

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QLabel, QLineEdit,
                             QListWidget, QPushButton, QTabWidget, QWidget,
                             QTableWidget, QTableView, QVBoxLayout)
from PyQt5.QtCore import QAbstractTableModel, Qt

import pandas as pd
import copy
import torch
import json
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from training.vrb_model import vrb_full_model
from gui.model_dataclasses import bounding_boxes
from training.rd_model import rd_full_model


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
        rd_model = None
    elif name == "vtranse":
        full_rd_model_kwargs = models_info[name]['full_rd_model_kwargs']
        rd_model, model = rd_full_model.create_rd_training_models(**full_rd_model_kwargs)

        rd_model_dict = torch.load(models_info[name]['rd_model_path'])
        rd_model.load_state_dict(rd_model_dict['model'])
        rd_model.eval()
        rd_model.to(device)
        model.to(device)

    return model, rd_model


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
                        'model': load_model(models_info, name)[0],
                        'rd_model': load_model(models_info, name)[1],
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

        self.objects_view = QTableView()
        self.objects_table = PandasQTableModel(obj_df, self.objects_view)
        self.objects_view.setModel(self.objects_table)
        self.objects_table.update_data(obj_df)
        self.objects_view.clicked.connect(self.object_clicked)

        self.relations_view = QTableView()
        self.relations_table = PandasQTableModel(relations_df, self.relations_view)
        self.relations_view.setModel(self.relations_table)
        self.relations_table.update_data(relations_df)
        self.relations_view.clicked.connect(self.relation_clicked)

        self.explanation_tab = ExplanationTab(self)

        self.tabs.addTab(self.objects_view, "Objects")
        self.tabs.addTab(self.relations_view, "Relations")
        self.tabs.addTab(self.explanation_tab, "Explanation")

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
        idx = int(item.row())
        chosen_row = self.boxes[idx]
        # TODO: do this better, don't make a deep copy...
        new_boxes = copy.deepcopy(self.boxes)
        new_boxes.detections = [chosen_row]

        img = self.parent.og_img
        self.parent.update_display_img(bounding_boxes.draw_boxes(img, new_boxes, threshold=0.))

        if self.parent.model['name'] == 'vtranse':
            import numpy as np
            from torchvision.transforms import functional as F

            def _create_box_image(box, img_shape) -> np.ndarray:
                mask = np.zeros(img_shape[0:2], dtype=np.uint8)
                xmin = int(np.round(box['xmin'].detach().cpu().numpy() * img_shape[1]))
                xmax = int(np.round(box['xmax'].detach().cpu().numpy() * img_shape[1]))
                ymin = int(np.round(box['ymin'].detach().cpu().numpy() * img_shape[0]))
                ymax = int(np.round(box['ymax'].detach().cpu().numpy() * img_shape[0]))
                mask[ymin:ymax, xmin:xmax] = 255
                return mask

            box_imgs = []
            labels = []
            obj_score = []
            for box in self.boxes:
                score = box['score']
                box_img = _create_box_image(box, self.parent.og_img.shape)
                box_imgs.append(box_img)
                labels.append(box['label'])
                obj_score.append(score)
            img = F.to_tensor(self.parent.og_img)
            losses, detections, features = self.parent.model['model']([img.to(self.parent.device)])

            sub = box_imgs[idx]
            sub = F.to_tensor(sub)
            relations = []

            rel_thresh = 0.3
            thresh = 0.3
            for j, box_img in enumerate(box_imgs):
                obj = box_img
                obj = F.to_tensor(obj)
                out = self.parent.model['rd_model'](features,
                                                    sub_inputs=[sub.to(self.parent.device)],
                                                    obj_inputs=[obj.to(self.parent.device)])
                for k, score in enumerate(out[0].detach().cpu().numpy()):
                    if score > rel_thresh and obj_score[j] > thresh:
                        rel = {"person_id": chosen_row['id'],
                               "obj_id": self.boxes[j]['id'],
                               "label": self.parent.model['relationships'][k],
                               "score": score}
                        relations.append(rel)
            sorted_relations = sorted(relations, key=lambda k: k['score'], reverse=True)
            self.boxes.relations = sorted_relations
            self.relations_table.update_data(self.boxes.relations_as_df())


class PandasQTableModel(QAbstractTableModel):

    def __init__(self, data, parent):
        QAbstractTableModel.__init__(self)
        self.parent = parent
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
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
        self.parent.resizeColumnsToContents()

class ExplanationTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.explanation_label = QLabel()
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setText("Confidence: "
                                       "A value in [0, 1] that can be interpreted as a"
                                       " percentage. Indicates how likely the indicated"
                                       " object class or relationship is correct. \n\n"
                                       "Risk: A percentage that indicates the likelihood"
                                       " that the given objects and relationships pose"
                                       " a safety and security risk.")
        layout.addWidget(self.explanation_label)
