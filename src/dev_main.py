

if __name__ == "__main__":
    import confuse
    from pathlib import Path
    from argparse import Namespace
    from training.data_util.vtranse_dataset import VTranseDataset, VTranseObjDataset, VTranseRelDataset
    from matplotlib import pyplot as plt
    import numpy as np

    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseRelDataset(dataset_path)

    # from rd_model import rd_full_model
