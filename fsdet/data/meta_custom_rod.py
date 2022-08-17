import contextlib
import io
import os
from PIL import Image

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

img_id = 100
# Returns in 'x0,y0,w,h' format
def bbox_from_line(line):
    annot = line.split(' ')[1:]
    annot = [float(i) for i in annot]
        
    return annot

def get_annotations_of_object(rel_path, obj, thing_classes, shots):
    if obj != 'Rod':
        global img_id
        data = []
        path = rel_path + obj + '/Label'
        files = os.listdir(path)
        
        counter = 0
        for file in files:
            if counter >= shots:
                break
            
            img_name = file.split('.')[0] + '.jpg'          # .txt -> .jpg
            img_path = rel_path + obj + '/' + img_name
            new_path = path + '/' + file

            img = Image.open(img_path)
            w,h = img.size
            img_id += 1
            
            annotation = {}
            annotation['file_name'] = img_path
            annotation['image_id'] = img_id
            annotation['height'] = h
            annotation['width'] = w
            
            f = open(new_path, 'r')
            instances = []
            for line in f.readlines():        
                bbox = bbox_from_line(line)
                
                instances.append( 
                    {
                        'category_id': thing_classes.index(obj.lower()),
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYWH_ABS
                    } 
                )
            
            annotation['annotations'] = instances
            data.append(annotation)
            counter += 1
            
        return data

    else:
        json_file = 'datasets/Custom-Rod/rod_background.json'
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]


        imgs_anns = list(zip(imgs, anns))
        ann_keys = ["iscrowd", "bbox", "category_id"]
        image_root = 'datasets/Custom-Rod'
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

def custom_rod_loader(name, load_classes):
    data = []
    classes = [i.title() for i in load_classes]             
    for obj in classes:
        data += get_annotations_of_object('datasets/custom_airplane/', obj, thing_classes, 4)
        
    return data

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
