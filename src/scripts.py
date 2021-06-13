import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
import confuse
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torchvision
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

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


def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(args, cfg):
    dataset_path = Path(cfg['visual_genome_path'].get())

    log_folder = Path(cfg['tensorboard_path'].get())
    writer = SummaryWriter(log_folder)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=201,
                                    trainable_backbone_layers=5)
    model.to(device)
    model.train()

    #  train all layerss
    for param in model.parameters():
        param.requires_grad = True

    modules = list(model.modules())

    # turn off grad on frozen backnorm layers, they were causing nan losses
    for module in modules:
        for param in module.parameters():
            if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d):
                param.requires_grad = False

    optimizer_kwargs = cfg['optimizer_kwargs'].get()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params, **optimizer_kwargs)

    ds = VTranseObjDataset(dataset_path)
    torch_ds = torch.utils.data.DataLoader(ds,
                                           batch_size=2, num_workers=8,
                                           collate_fn=collate_fn)

    num_epochs = cfg['num_epochs'].get()
    step_count = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        
        for images, targets in tqdm(torch_ds):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(images, targets)

            losses = sum(loss for loss in out.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            for loss_key in out.keys():
                writer.add_scalar(loss_key, out[loss_key], step_count)
            step_count += 1

        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(epoch).zfill(3) + "resnet50_fpn_frcnn_full.tar")
