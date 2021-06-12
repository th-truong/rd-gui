import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
import confuse


from gui.main_window import MainWindow
from training.data_util.visual_genome_dataset import VGRelationsDataset
from training.data_util.vtranse_dataset import VTranseDataset, VTranseObjDataset


def display_gui(args):
    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    app = QApplication(sys.argv)

    window = MainWindow(cfg)
    window.show()

    sys.exit(app.exec_())


def train_model(args, cfg):
    dataset_path = Path(cfg['visual_genome_path'].get())
    return VTranseObjDataset(dataset_path)
