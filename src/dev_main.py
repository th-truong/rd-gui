def collate_fn(batch):
    return tuple(zip(*batch))

def count_relationships():
    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseRelTrainDataset(dataset_path)

    torch_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    rel_counter = np.zeros(100)
    for _ in tqdm(torch_ds):
        img, targets, sub_inputs, obj_inputs = _
        for i in range(len(targets[0]['rel_labels'])):
            classes = targets[0]['rel_labels'].numpy()
            for idx in np.nonzero(classes):
                rel_counter[idx] += 1
    return rel_counter


def calculate_rel_percentage(rel_id, rel_counter, percentage_count=27174.0):
    # percentage_count = 27174 is np.median(rel_counter), so 50% of the id's will be subsampled
    if random.random() < percentage_count / rel_counter[rel_id]:
        return True
    else:
        return False


def train_rel_model():
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

    rel_counter = np.load("relationship_counts_train.npy")


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
                rel_id = np.argmax(targets[0]['rel_labels'][i].cpu().numpy())
                if calculate_rel_percentage(rel_id=rel_id, rel_counter=rel_counter):
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


def test_image():
    from PIL import Image
    from torchvision.transforms import functional as F
    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))
    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseRelTrainDataset(dataset_path, ds_set="test")
    torch_ds = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    full_rd_model_kwargs = cfg['full_rd_model_kwargs'].get()
    model, mrcnn_model = rd_full_model.create_rd_training_models(**full_rd_model_kwargs)
    model.eval()
    mrcnn_model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mrcnn_model.to(device)
    model.to(device)

    saved_model = torch.load(r"D:\paper_repos\rd-gui\src\pretrained_models\first_rel_model\79_8733914_full_epoch.tar")

    model.load_state_dict(saved_model['model'])

    img_path = r"D:\paper_repos\rd-gui\src\142322083_bab79bb28d_o.jpg"
    img_np = np.array(Image.open(img_path))
    img = F.to_tensor(img_np)
    losses, detections, features = mrcnn_model([img.to(device)])
    detections = detections[0]
    thresh = 0.0

    def _create_box_image(box_coords, img_shape) -> np.ndarray:
        mask = np.zeros(img_shape[0:2], dtype=np.uint8)
        mask[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]] = 255
        return mask

    boxes = []
    labels = []
    scores = []
    for i, score in enumerate(detections['scores']):
        if score > thresh:
            coords = [int(np.round(x)) for x in detections['boxes'][i].cpu().detach().numpy()]
            box = _create_box_image(coords, img_np.shape)
            boxes.append(box)
            labels.append(ds.id2obj[int(detections['labels'][i].cpu().detach().numpy())])
            scores.append(detections['scores'][i])

    pair_indices = [3, 6]

    sub = boxes[pair_indices[0]]
    sub = F.to_tensor(sub)
    obj = boxes[pair_indices[1]]
    obj = F.to_tensor(obj)

    out = model(features, sub_inputs=[sub.to(device)], obj_inputs=[obj.to(device)])

    return img_np, boxes, labels, scores, out.cpu().detach().numpy(), ds

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
    #         break

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import imquality.brisque as brisque
from functools import partial
from multiprocessing import Pool, Manager
import json

def calculate_brisque(brisque_score_dict, image_id):
    img = Image.open(Path(r'D:/datasets/visual_genome') / "VG_100K" / (image_id + ".jpg"))
    score = brisque.score(img)
    brisque_score_dict[image_id] = score

def calc_brisque_scores():
    with open("image_ids.json", 'r') as f:
        ds_idxs = json.load(f)

    ds_idxs = ds_idxs

    n_workers = 10

    with Manager() as manager:
        brisque_score_dict = manager.dict()
        # make a partial which always feeds the manager dictionary
        calculate_brisque_partial = partial(calculate_brisque, brisque_score_dict)
        with Pool(processes=n_workers) as pool:
            iterators = tqdm(pool.imap(calculate_brisque_partial, ds_idxs), total=len(ds_idxs))
            # iterate through
            tuple(iterators)

        with open('brisque_scores.json', 'w') as fp:
            json.dump(brisque_score_dict.copy(), fp)

    # ['2323812', '2349720', '2405985'] are the broken image ids

def clean_brisque_score():
    # this one does not have
    with open("brisque_scores.json", 'r') as fp:
        brisque_scores = json.load(fp)
    brisque_scores_cleaned = {}
    for image_id, score in brisque_scores.items():
        if score is not None:
            if 0 <= score <= 100:
                brisque_scores_cleaned[image_id] = score
    with open('brisque_scores_cleaned.json', 'w') as fp:
        json.dump(brisque_scores_cleaned, fp)


def analyze_brisque_scores():
    import numpy as np
    from matplotlib import pyplot as plt
    with open("brisque_scores_cleaned.json", 'r') as fp:
        brisque_scores = json.load(fp)
    histogram = plt.hist([x for x in brisque_scores.values()], bins=[i for i in range(0, 101, 5)])

    # the final 9 values, 60-100, can be clumped together to make sure each bin has enoughesamples
    np.sum(histogram[0][-9:])

    histogram_even = plt.hist([x for x in brisque_scores.values()], bins=[i for i in range(0, 61, 5)]+[100])


def get_obj_predictions():
    import sys
    from pathlib import Path
    from PyQt5.QtWidgets import QApplication
    from argparse import Namespace
    import confuse
    from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
    import torchvision
    import torch
    from tqdm import tqdm
    import pickle

    from torch.utils.tensorboard import SummaryWriter

    from gui.main_window import MainWindow
    from training.data_util.visual_genome_dataset import VGRelationsDataset
    from training.data_util.vtranse_dataset import VTranseDataset, VTranseObjDataset

    import numpy as np
    from torchvision.transforms import functional as F

    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=201,
                                    trainable_backbone_layers=5)
    model.to(device)
    model.eval()

    model_dict = torch.load(r"D:\paper_repos\rd-gui\src\pretrained_models\frcnn_vtranse\039resnet50_fpn_frcnn_full.tar")
    model.load_state_dict(model_dict['model'])

    ds = VTranseObjDataset(dataset_path, ds_set="test")
    obj_predictions = {image_id: None for image_id in ds.ds_idxs}

    with torch.no_grad():
        for i, ds_out in tqdm(enumerate(ds), total=len(ds)):
            image, target = ds_out
            image_id = ds.ds_idxs[i]
            images = list([image.to(device)])
            targets = [{k: v.to(device) for k, v in target.items()}]

            out = model(images, targets)

            boxes = out[0]['boxes'].cpu().detach().numpy().squeeze()
            scores = out[0]['scores'].cpu().detach().numpy().squeeze()
            labels = out[0]['labels'].cpu().detach().numpy().squeeze()

            obj_predictions[image_id] = {'boxes': boxes,
                                         'scores': scores,
                                         'labels': labels}
    with open("obj_predictions.p", 'wb') as f:
        pickle.dump(obj_predictions, f)


if __name__ == "__main__":
    # for training
    # train_rel_model()

    # for counting rels
    # rel_counter = count_relationships()

    # for calculating and sving brisque scores
    # calc_brisque_scores()

    # for testing output of model
    # img_np, boxes, labels, scores, out, ds = test_image()

    # apples = np.argsort(out[0])

    # print(ds.id2predicate[int(apples[-1])])
    # print(out[0][apples[-1]])
    # print(ds.id2predicate[int(apples[-2])])
    # print(out[0][apples[-2]])
    # print(ds.id2predicate[int(apples[-3])])
    # print(out[0][apples[-3]])
    import enum
    import pickle
    import numpy as np
    import mapcalc
    from training.data_util.vtranse_dataset import VTranseObjDataset
    from argparse import Namespace
    import confuse

    with open("obj_rel_detections/obj_predictions.p", "rb") as f:
        obj_pred = pickle.load(f)

    args = Namespace(cfg_path='cfgs/full_model_training_config.yml')

    cfg = confuse.Configuration('RD_GUI', __name__, read=False)
    cfg.set_file(Path(args.cfg_path))

    dataset_path = Path(cfg['visual_genome_path'].get())

    ds = VTranseObjDataset(dataset_path, ds_set="test")
