import confuse
from pathlib import Path
from argparse import Namespace
from training.data_util.vtranse_dataset import VTranseDataset, VTranseObjDataset, VTranseRelDataset, VTranseRelTrainDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

from training.rd_model import rd_full_model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_rel_model():
    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseRelTrainDataset(dataset_path)

    torch_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    full_rd_model_kwargs = cfg['full_rd_model_kwargs'].get()
    model, mrcnn_model = rd_full_model.create_rd_training_models(**full_rd_model_kwargs)
    mrcnn_model.eval()
    model.train()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mrcnn_model.to(device)
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer_kwargs'].get())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg['scheduler_kwargs'].get())

    log_folder = Path(cfg['tensorboard_path'].get())
    writer = SummaryWriter(log_folder)


    # 803276 total relationships in trianing set
    step = 0
    for epoch in range(cfg['num_epochs'].get()):
        print(f"{epoch=}")
        for _ in tqdm(torch_ds):
            img, targets, sub_inputs, obj_inputs = _
            img = [x.to(device) for x in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # iterates through the tuple for batch size and then the list to convert todevice

            for i, y in enumerate(sub_inputs):
                for j, x in enumerate(y):
                    sub_inputs[i][j] = sub_inputs[i][j].to(device)
            for i, y in enumerate(obj_inputs):
                for j, x in enumerate(y):
                    obj_inputs[i][j] = obj_inputs[i][j].to(device)

            num_correct = 0
            total_labels = 0
            with torch.no_grad():
                losses, detections, features = mrcnn_model(img)
            for i in range(len(targets[0]['rel_labels'])):
                # hacky subsampling for now...
                # TODO: calculate proper subsampling or maybe use pos_weights
                # skip 95% of these classes, imbalance is much worse than that it seems though
                
                if (np.argmax(targets[0]['rel_labels'][i].cpu().numpy()) == 54
                    or np.argmax(targets[0]['rel_labels'][i].cpu().numpy()) == 96
                    or np.argmax(targets[0]['rel_labels'][i].cpu().numpy()) == 42
                    or np.argmax(targets[0]['rel_labels'][i].cpu().numpy()) == 38
                    or np.argmax(targets[0]['rel_labels'][i].cpu().numpy()) == 53) and random.random() > 0.05:
                    pass
                else:
                    # careful with inputs, sub_inputs and obj_inputs should be a list. if slicing remove the parenthesese
                    out = model(features, targets, [sub_inputs[0][i]], [obj_inputs[0][i]])

                    losses = loss_fn(out.squeeze(), targets[0]['rel_labels'][i])

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    if np.argmax(out.squeeze().cpu().detach().numpy()) == np.argmax(targets[0]['rel_labels'][i].cpu().numpy()):
                        num_correct += 1
                    step += 1
                    total_labels += 1
            writer.add_scalar("last_prediction", np.argmax(out.squeeze().cpu().detach().numpy()), step)
            if total_labels > 0: writer.add_scalar("percentage", num_correct/total_labels, step)
        save_path = Path(cfg['tensorboard_path'].get()) / (str(epoch+1) + "_" + str(step) + "_full_epoch.tar")
        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "global_step": step},
                    save_path)
        scheduler.step()

    # plt.imshow(out[0][0][0].squeeze())
    #out = model(img)


if __name__ == "__main__":
    train_rel_model()
    # args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    # cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    # cfg.set_file(Path(args.cfg_path))
    # dataset_path = Path(cfg['visual_genome_path'].get())

    # ds = VTranseRelTrainDataset(dataset_path, ds_set="test")
    # torch_ds = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # full_rd_model_kwargs = cfg['full_rd_model_kwargs'].get()
    # model, mrcnn_model = rd_full_model.create_rd_training_models(**full_rd_model_kwargs)
    # model.eval()
    # mrcnn_model.eval()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # mrcnn_model.to(device)
    # model.to(device)

    # saved_model = torch.load(r"D:\paper_repos\rd-gui\src\pretrained_models\first_rel_model\1_660944_full_epoch.tar")

    # model.load_state_dict(saved_model['model'])
    # for _ in tqdm(torch_ds):
    #     img, targets, sub_inputs, obj_inputs = _
    #     img = [x.to(device) for x in img]
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     # iterates through the tuple for batch size and then the list to convert todevice

    #     for i, y in enumerate(sub_inputs):
    #         for j, x in enumerate(y):
    #             sub_inputs[i][j] = sub_inputs[i][j].to(device)
    #     for i, y in enumerate(obj_inputs):
    #         for j, x in enumerate(y):
    #             obj_inputs[i][j] = obj_inputs[i][j].to(device)
    #     with torch.no_grad():
    #         losses, detections, features = mrcnn_model(img)
    #     for i in range(len(targets[0]['rel_labels'])):
    #         # careful with inputs, sub_inputs and obj_inputs should be a list. if slicing remove the parenthesese
    #         out = model(features, targets, [sub_inputs[0][i]], [obj_inputs[0][i]])

    #         print("-----------------------------------------")
    #         print("preds")
    #         print(np.argmax(out.cpu().detach().numpy()))
    #         print(np.max(out.cpu().detach().numpy()))
    #         print("true")
    #         print(np.argmax(targets[0]['rel_labels'][i].cpu().numpy()))
    #         print(np.max(targets[0]['rel_labels'][i].cpu().numpy()))
    #         plt.imshow(np.moveaxis(img[0].cpu().squeeze().numpy(), 0, -1))
    #     break
