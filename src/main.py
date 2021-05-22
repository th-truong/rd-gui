import sys
from pathlib import Path
import argparse
from PyQt5.QtWidgets import QApplication
import confuse
from pathlib import Path

from gui.main_window import MainWindow


if __name__ == "__main__":
    # ideas for functionality:
    # 3. relationship tree -> select a person and a tree pops up with all relationships
    # 2. identifying with hand is holding the objects -> maybe use a pose detector to do this?
    # 1. user parameters for choosing certain parameters on plots, risk of error, etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', default='cfgs/config.yml')

    args = parser.parse_args()

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    app = QApplication(sys.argv)

    window = MainWindow(cfg)
    window.show()

    sys.exit(app.exec_())
