import contextlib
import io
import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

thing_classes = ['rod']
base_classes = []
novel_classes = ['rod'] 

metadata = {
    "thing_classes": thing_classes,
    "base_classes": base_classes,
    "novel_classes": novel_classes
}

def custom_rod_loader(name, load_classes):
    json_file = 'datasets/custom-rod/rod.json'
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    img_ids = sorted(list(coco_api.imgs.keys()))
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]


    imgs_anns = list(zip(imgs, anns))
    ann_keys = ["iscrowd", "bbox", "category_id"]
    image_root = 'datasets/custom-rod'
    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(
            image_root, img_dict["file_name"]
        )
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            objs.append(obj)
        record["annotations"] = objs
    
        dataset_dicts.append(record)

    return dataset_dicts

def register_custom_rod(name, thing_classes, metadata):
    # register dataset (step 1)
    DatasetCatalog.register(
        name, # name of dataset, this will be used in the config file
        lambda: custom_rod_loader( # this calls your dataset loader to get the data
            name, thing_classes, # inputs to your dataset loader
        ),
    )

    # register meta information (step 2)
    MetadataCatalog.get(name).set(
        thing_classes=metadata["thing_classes"], # all classes
        base_classes=metadata["base_classes"], # base classes
        novel_classes=metadata["novel_classes"], # novel classes
    )
    MetadataCatalog.get(name).evaluator_type = "custom_rod" # set evaluator
