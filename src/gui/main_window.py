from PyQt5.QtWidgets import QMainWindow, QGridLayout, QLabel, QPushButton, QTabWidget, QWidget
from PyQt5.QtGui import QFont

from gui.image_tab import ImageTab

import pandas as pd


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()

        self.title = 'Object Relationship Detection Tool'
        self.left = 0
        self.top = 0
        self.width = 1280
        self.height = 720
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        font = self.font()
        font.setPointSize(18)
        self.setFont(font)

        self.tab_widget = Tabs(self, cfg)
        self.setCentralWidget(self.tab_widget)

        obj_df = pd.DataFrame({'Object': [],
                               'Confidence': []})
        relations_df = pd.DataFrame({'Relationship': [],
                                     'Confidence': []})
        self.tab_widget.image_tab.results_tables.objects_table.update_data(obj_df)
        self.tab_widget.image_tab.results_tables.relations_table.update_data(relations_df)

    def closeEvent(self, event):
        pass


class Tabs(QWidget):
    def __init__(self, parent, cfg):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.image_tab = ImageTab(self, cfg)

        self.tabs.addTab(self.image_tab, "Images")

        layout = QGridLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # def wheelEvent(self, event: QWheelEvent):  # for mosue scrolls later
    #     tab = self.tabs.currentWidget()
    #     if event.angleDelta().y() > 0:
    #         tab.fwd_btn_click()
    #     elif event.angleDelta().y() < 0:
    #         tab.back_btn_click()
