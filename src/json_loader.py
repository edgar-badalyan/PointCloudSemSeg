import json
import os
import numpy as np
import skimage.io as io
from skimage.color import gray2rgb
from typing import List
from dataset_loader import DatasetLoader

# Detectron2
from detectron2.structures import BoxMode


class JSONDatasetLoader(DatasetLoader):

    def __init__(self, dataset_dir: str):
        """
        Constructor.
        Parameters
        ----------
        dataset_dir: str
        Path to the dataset.
        """
        super().__init__(dataset_dir)
        self.DATASET_DIR = dataset_dir

        self.IMAGE_DIR = os.path.join(self.DATASET_DIR, "images")
        self.ANNOTATION_DIR = os.path.join(self.DATASET_DIR, "annotations")

    def load_image(self, im_name: str, im_dir: str = None) -> np.ndarray:
        """
        Loads an image from the dataset. The models work by default with RGB images, so if the image
        is grayscale, the pixel value will tripled to cover all three channels.
        Parameters
        ----------
        im_name: str
        Name of the image.
        im_dir: str
        Subdirectory in which to find the image (e.g., 'test', or 'train').
        If unspecified, it will be assumed that the image is in the root directory.
        Returns
        -------
        np.ndarray: the image.
        """
        if im_dir is not None:
            im_name = f"{self.IMAGE_DIR}/{im_dir}/{im_name}"
        else:
            im_name = f"{self.IMAGE_DIR}/{im_name}"

        im = io.imread(im_name)
        if len(im.shape) < 3:
            im = gray2rgb(im)
        return im

    def get_window_dicts(self, im_dir: str) -> List[dict]:
        """
        Creates a dataset with the format specified by detecton2:
        see https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
        Parameters
        ----------
        im_dir: str
        Subdirectory to register in the dataset.

        Returns
        -------
        List[dict]: a list with each dict representing an image in the dataset
        """
        im_dir = os.path.join(self.IMAGE_DIR, im_dir)

        files = os.listdir(im_dir)

        dataset_dicts = []
        for file in files:

            im_id = file.split('.')[0]
            
            anno_data = self.parse_annotation(f"{im_id}.json")

            im_name_abs = f"{im_dir}/{file}"
            anno_data["file_name"] = im_name_abs
            anno_data["image_id"] = im_id
          
            objs = self.annotation_detectron_format(anno_data)
            
            anno_data["annotations"] = objs
            dataset_dicts.append(anno_data)
            
        return dataset_dicts
        
    def parse_annotation(self, name: str) -> dict:
        """
        Read an annotation file and returns the relevant contents in a dict.
        Parameters
        ----------
        name: str
        Name of the instance for which to read the annotation

        Returns
        -------
        dict: dict with file name, image height, image width, and list of annotated regions in the image.
        """
        f = open(f"{self.ANNOTATION_DIR}/{name}")
        data = json.load(f)
        
        anno_data = {}
        regions = []
        
        for region in data["regions"]:
            regions.append((region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]))
        
        anno_data["regions"] = regions
        anno_data["file_name"] = data["filename"]
        anno_data["width"] = data["width"]
        anno_data["height"] = data["height"]
        
        return anno_data
        
    @staticmethod
    def annotation_detectron_format(anno_data: dict) -> List[dict]:
        """
        Converts the polygons in the annotation to the format required by detectron2.
        Parameters
        ----------
        anno_data: dict
        The annotation dict

        Returns
        -------
        List[dict]: a list where each element represents a polygon in the image
        """
        annos = anno_data["regions"]
            
        objs = []
        for region in annos:
            px = region[0]
            py = region[1]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }

            objs.append(obj)
            
        return objs
