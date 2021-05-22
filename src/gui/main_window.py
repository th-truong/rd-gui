from PyQt5.QtWidgets import QMainWindow, QGridLayout, QLabel, QPushButton, QTabWidget, QWidget

from gui.image_tab import ImageTab

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

        self.tab_widget = Tabs(self, cfg)
        self.setCentralWidget(self.tab_widget)

    def closeEvent(self, event):
        pass


class Tabs(QWidget):
    def __init__(self, parent, cfg):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        image_tab = ImageTab(self, cfg)

        self.tabs.addTab(image_tab, "Images")

        layout = QGridLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # def wheelEvent(self, event: QWheelEvent):  # for mosue scrolls later
    #     tab = self.tabs.currentWidget()
    #     if event.angleDelta().y() > 0:
    #         tab.fwd_btn_click()
    #     elif event.angleDelta().y() < 0:
    #         tab.back_btn_click()