import pandas as pd
from pathlib import Path
from tqdm import tqdm


def save_object_metadata():
    objects_df = pd.read_pickle(r"D:\datasets\visual_genome\objects.pkl")
    total_objs = []

    for i, row in tqdm(objects_df.iterrows()):
        img_id = row.image_id
        objs = row.objects
        for obj in objs:
            obj_dict = {"image_id": img_id,
                        "name": obj['names'][0],
                        'x': obj['x'],
                        'y': obj['y'],
                        'w': obj['w'],
                        'h': obj['h']}
            total_objs.append(obj_dict)
    total_objs_df = pd.DataFrame(total_objs)
    total_objs_df.to_pickle(r"D:\datasets\visual_genome\metadata\objects_metadata.pkl")


if __name__ == '__main__':
    save_object_metadata()
    # TODO: look  at the vtranse h5 file for the objects and predicates to be used.
