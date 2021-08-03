

def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
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

    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseRelTrainDataset(dataset_path)

    torch_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    from training.rd_model import rd_full_model

    full_rd_model_kwargs = cfg['full_rd_model_kwargs'].get()
    model, mrcnn_model = rd_full_model.create_rd_training_models(**full_rd_model_kwargs)
    mrcnn_model.eval()
    model.train()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mrcnn_model.to(device)
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer_kwargs'].get())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg['scheduler_kwargs'].get())

    log_folder = Path(cfg['tensorboard_path'].get())
    writer = SummaryWriter(log_folder)


    # 803276 total relationships in trianing set
    step = 0
    for epoch in range(cfg['num_epochs'].get()):
        print(f"{epoch=}")
        for _ in torch_ds:
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

            with torch.no_grad():
                losses, detections, features = mrcnn_model(img)

            for i in range(len(targets[0]['rel_labels'])):
                # careful with inputs, sub_inputs and obj_inputs should be a list. if slicing remove the parenthesese
                out = model(features, targets, [sub_inputs[0][i]], [obj_inputs[0][i]])

                losses = loss_fn(out.squeeze(), targets[0]['rel_labels'][i])
                writer.add_scalar("train_loss", losses, step)
                step += 1

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
        scheduler.step()

    # plt.imshow(out[0][0][0].squeeze())
    #out = model(img)
