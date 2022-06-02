import fiftyone as fo
import fiftyone.zoo as foz
import os
# Download and load the validation split of COCO-2017
# dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# session = fo.launch_app(dataset)


# # # Import the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path='dataset/voc2coco/val2017' ,
#     labels_path="dataset/voc2coco/annotations/instances_val2017.json",
# )
# dataset.name = "coco2"
# # session = fo.launch_app(dataset)
# # session.wait()
dataset = foz.load_zoo_dataset("coco-2017", split="validation")
DATASET_PATH = "val2017_bbox_results.json"
data_dir = os.path.dirname(dataset.first().filepath)
print(data_dir)
predictions = fo.Dataset.from_json(DATASET_PATH, rel_dir=data_dir)
dataset.merge_samples(predictions)
session = fo.launch_app(dataset)
session.wait()
