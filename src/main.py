import sys
from pathlib import Path
import argparse
from PyQt5.QtWidgets import QApplication
import confuse
from pathlib import Path

from gui.main_window import MainWindow


def display_gui(args):
    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    app = QApplication(sys.argv)

    window = MainWindow(cfg)
    window.show()

    sys.exit(app.exec_())


def train_model(args):
    pass


if __name__ == "__main__":
    # ideas for functionality:
    # 3. relationship tree -> select a person and a tree pops up with all relationships
    # 2. identifying with hand is holding the objects -> maybe use a pose detector to do this?
    # 1. user parameters for choosing certain parameters on plots, risk of error, etc.
    # 0. list of deep learning models used for inference
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gui = subparsers.add_parser('gui')
    parser_gui.set_defaults(func=display_gui)
    parser_gui.add_argument('--cfg-path', default='cfgs/gui_config.yml')

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_model)
    parser_train.add_argument('--cfg-path', default='cfgs/training_config.yml')

    args = parser.parse_args()
    args.func(args)