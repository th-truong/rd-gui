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


def display_gui(args, cfg):

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

    model_dict = torch.load(r"D:\paper_repos\rd-gui\src\039resnet50_fpn_frcnn_full.tar")
    model.load_state_dict(model_dict['model'])

    optimizer_kwargs = cfg['optimizer_kwargs'].get()
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=2,
    #                                                gamma=0.1)
    optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[10, 20],
                                                        gamma=0.1)


    ds = VTranseObjDataset(dataset_path)
    torch_ds = torch.utils.data.DataLoader(ds,
                                           batch_size=2, num_workers=8,
                                           collate_fn=collate_fn)

    num_epochs = cfg['num_epochs'].get()
    step_count = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        num_correct = 0
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
        lr_scheduler.step()
        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(epoch).zfill(3) + "resnet50_fpn_frcnn_full.tar")
