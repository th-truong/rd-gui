import h5py
from tqdm import tqdm
import json
import pickle
import numpy as np


if __name__ == '__main__':
    ds_h5 = h5py.File(r"D:\datasets\visual_genome\vg1_2_meta.h5")

    train_ds = {}
    test_ds = {}

    for key in tqdm(ds_h5['gt']['train'].keys()):
        sample_dict = {'sub_boxes': np.array(ds_h5['gt']['train'][key]['sub_boxes']).tolist(),
                       'rlp_labels': np.array(ds_h5['gt']['train'][key]['rlp_labels']).tolist(),
                       'obj_boxes': np.array(ds_h5['gt']['train'][key]['obj_boxes']).tolist()}
        train_ds[key] = sample_dict

    for key in tqdm(ds_h5['gt']['test'].keys()):
        sample_dict = {'sub_boxes': np.array(ds_h5['gt']['test'][key]['sub_boxes']).tolist(),
                       'rlp_labels': np.array(ds_h5['gt']['test'][key]['rlp_labels']).tolist(),
                       'obj_boxes': np.array(ds_h5['gt']['test'][key]['obj_boxes']).tolist()}
        test_ds[key] = sample_dict


    pickle_out = open(r"D:\paper_repos\rd-gui\src\training\data_util\train.pkl","wb")
    pickle.dump(train_ds, pickle_out)
    pickle_out.close()

    pickle_out = open(r"D:\paper_repos\rd-gui\src\training\data_util\test.pkl","wb")
    pickle.dump(test_ds, pickle_out)
    pickle_out.close()