import os
from PIL import Image

from detectron2.config import global_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from fsdet.evaluation.evaluator import DatasetEvaluator

thing_classes = ['airplane', 'car', 'cat', 'dog']
base_classes = ['car', 'cat', 'dog']
novel_classes = ['airplane'] 

metadata = {
    "thing_classes": thing_classes,
    "base_classes": base_classes,
    "novel_classes": novel_classes
}

img_id = 0

# TODO Přidat 'iscrowd' kompatabilitu
# Returns in 'x0,y0,w,h' format
def bbox_from_line(line):
    annot = line.split(' ')[1:]
    annot = [float(i) for i in annot]
        
    return annot

def get_annotations_of_object(rel_path, obj, thing_classes, shots):
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

# TODO Tady případně dát if '"shots" in name...', ať to má možnost mít obecně k-shot (teď je automaticky 10-shot)
def custom_airplane_loader(name, load_classes):
    if name == 'custom_airplane_all':
        shots = 10
    else:
        shots = float('inf')
    data = []
    classes = [i.title() for i in load_classes]             
    for obj in classes:
        data += get_annotations_of_object('datasets/custom_airplane/', obj, thing_classes, shots)
        
    return data


# class CustomAirplaneEvaluator(DatasetEvaluator):
#     def __init__(self, dataset_name): # initial needed variables
#         self._dataset_name = dataset_name

#     def reset(self):            # reset predictions
#         self._predictions = []

#     def process(self, inputs, outputs): # prepare predictions for evaluation
#         for input, output in zip(inputs, outputs):
#             prediction = {"image_id": input["image_id"]}
#             if "instances" in output:
#                 prediction["instances"] = output["instances"]
#             self._predictions.append(prediction)

#     def evaluate(self): # evaluate predictions
#         results = evaluate_predictions(self._predictions)
#         return {
#             "AP": results["AP"],
#             "AP50": results["AP50"],
#             "AP75": results["AP75"],
#         }


def register_custom_airplane(name, thing_classes, metadata):
    # register dataset (step 1)
    DatasetCatalog.register(
        name, # name of dataset, this will be used in the config file
        lambda: custom_airplane_loader( # this calls your dataset loader to get the data
            name, thing_classes, # inputs to your dataset loader
        ),
    )

    # register meta information (step 2)
    MetadataCatalog.get(name).set(
        thing_classes=metadata["thing_classes"], # all classes
        base_classes=metadata["base_classes"], # base classes
        novel_classes=metadata["novel_classes"], # novel classes
    )
    MetadataCatalog.get(name).evaluator_type = "custom_airplane" # set evaluator
