import argparse
import confuse
from pathlib import Path
from scripts import display_gui, train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gui = subparsers.add_parser('gui')
    parser_gui.set_defaults(func=display_gui)
    parser_gui.add_argument('--cfg-path', default='cfgs/gui_config.yml')

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_model)
    parser_train.add_argument('--cfg-path', default='cfgs/training_config.yml')

    args = parser.parse_args()

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    args.func(args, cfg)
