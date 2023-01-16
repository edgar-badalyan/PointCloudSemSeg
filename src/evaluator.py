import numpy as np
import skimage
import json
import os
from typing import Union, Tuple, List

# Personal scripts
from dataset_loader import DatasetLoader
from custom_predictor import Predictor
import project_cubemap


# Detectron2
from detectron2.engine.defaults import DefaultPredictor


class ModelEvaluator:
    """
    Class providing methods to perform inference and evaluation of a model
    """

    def __init__(self, dataset_loader: DatasetLoader, predictor: Union[DefaultPredictor, Predictor]):
        """
        Constructor.
        Parameters
        ----------
        dataset_loader: DatasetLoader
        Object used to load images and annotation from the dataset
        predictor: Union[DefaultPredictor, Predictor]
        predictor
        """
        self.dataset_loader = dataset_loader
        self.predictor = predictor

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """
        Use the model to predict the mask of a single image.
        Parameters
        ----------
        image: np.ndarray
        Image on which to perform prediction.
        Returns
        -------
        np.ndarray: binary mask
        """
        outputs = self.predictor(image)
        pred = outputs["instances"].pred_masks

        mask_pred = pred.detach().cpu().numpy()
        mask_pred = np.any(mask_pred, axis=0)

        return mask_pred

    def predict_image_crop(self, im: np.ndarray, crop_size: Tuple[int, int], overlap: int = 0) -> np.ndarray:
        """
        Use the model to predict the mask of a single image by tiling it.
        Parameters
        ----------
        im: np.ndarray
        Image on which to perform prediction.
        crop_size: Tuple[int, int]
        Size of the tiles (in pixels)
        overlap: int
        Amount of overlap between tiles. Default is 0.

        Returns
        -------
        np.ndarray: binary mask
        """
        H, W = im.shape[0], im.shape[1]
        mask_pred = np.zeros([H, W], dtype=bool)

        y_max = H // crop_size[1]
        x_max = W // crop_size[0]

        for y in range(y_max + 1):
            py_min = y * crop_size[1]
            py_max = min((y + 1) * crop_size[1] + overlap, H)

            for x in range(x_max + 1):
                px_min = x * crop_size[0]
                px_max = min((x + 1) * crop_size[0] + overlap, W)

                im_crop = im[py_min:py_max, px_min:px_max]

                mask_pred_crop = self.predict_image(im_crop)

                mask_pred[py_min:py_max, px_min:px_max] = np.logical_or(mask_pred[py_min:py_max, px_min:px_max],
                                                                        mask_pred_crop)

        return mask_pred

    @staticmethod
    def format_annotations(annotation: List[dict]) -> np.ndarray:
        """
        Formats annotations in a way more suitable to making a mask.
        Parameters
        ----------
        annotation: List[dict]
        List of annotations.
        Returns
        -------
        Reformatted annotations.
        """
        polygons = []

        for poly in annotation:
            segm = poly["segmentation"][0]
            px = segm[::2]
            px = [[x] for x in px]
            py = segm[1::2]
            py = [[y] for y in py]

            polygon = np.hstack((px, py))
            polygons.append(polygon)

        return polygons

    def ground_truth_mask(self, annotation: List[dict], H:int, W:int) -> np.ndarray:
        """
        Make a binary mask from the annotations.
        Parameters
        ----------
        annotation: List[dict]
        List of annotations.
        H: image height
        W: image width

        Returns
        -------
        np.ndarray: binary mask
        """
        polygons = self.format_annotations(annotation)

        mask_true = np.zeros([H, W], dtype=bool)

        for poly in polygons:
            # print(poly)
            # segm = poly[0]

            px = poly[:, 0]
            py = poly[:, 1]

            rr, cc = skimage.draw.polygon(py, px, [H, W])
            mask_true[rr, cc] = 1

        return mask_true

    @staticmethod
    def compute_metrics(mask_true: np.ndarray, mask_pred: np.ndarray, verbose: bool = False) -> dict:
        """
        Compute precision, recall, and F1-score given ground truth and prediction
        Parameters
        ----------
        mask_true: np.ndarray
        Ground truth binary mask
        mask_pred: np.ndarray
        Prediction binary mask
        verbose: bool
        Whether to print the results before returning them. Default is False.

        Returns
        -------

        """
        true_positive = np.sum(mask_true * mask_pred)

        relevant = np.sum(mask_true)
        retrieved = np.sum(mask_pred)

        precision = true_positive / retrieved
        recall = true_positive / relevant
        F1_score = (2 * true_positive) / (relevant + retrieved)

        # Convert Nan to 0
        precision = precision if precision == precision else 0
        recall = recall if recall == recall else 0
        F1_score = F1_score if F1_score == F1_score else 0

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": F1_score
        }

        if verbose:
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"f1-score: {F1_score}")

        return metrics

    def evaluate_image(self, im: np.ndarray, anno: List[dict], crop_size: Tuple[int, int],
                       overlap: int) -> Tuple[dict, np.ndarray]:
        """
        Evaluate metrics on the prediction of a single image
        Parameters
        ----------
        im: np.ndarray
        Image.
        anno: List[dict]
        Annotations.
        crop_size: Tuple[int, int]
        Size of the tiles (in pixels)
        overlap: int
        Amount of overlap between tiles. A warning is printed if
        crop_size is None but overlap is different from 0

        Returns
        -------
        Tuple[dict, np.ndarray]: metrics and predicted mask
        """

        if crop_size is None:
            if overlap != 0:
                print("Warning: overlap is set but crop is None, overlap will be ignored.")

            mask_pred = self.predict_image(im)
        else:
            mask_pred = self.predict_image_crop(im, crop_size, overlap)

        H, W = im.shape[0], im.shape[1]

        mask_true = self.ground_truth_mask(anno, H, W)

        metrics = self.compute_metrics(mask_true, mask_pred, verbose=False)

        return metrics, mask_pred

    def evaluate_dataset(self, im_dir: str, log_dir: str, crop_size: Tuple[int, int], overlap: int):
        """
        Evaluate predictions on a dataset.
        Parameters
        ----------
        im_dir: str
        Subdir of the dataset to evaluate on (e.g., 'test', 'val').
        log_dir: str
        Path to logging directory
        crop_size: Tuple[int, int]
        Size of the tiles (in pixels)
        overlap: int
        Amount of overlap between tiles. A warning is printed if
        crop_size is None but overlap is different from 0

        Returns
        -------

        """

        dataset_dicts = self.dataset_loader.get_window_dicts(im_dir)

        os.makedirs(log_dir, exist_ok=True)

        metrics_all = {}

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0

        for d in dataset_dicts:
            im = self.dataset_loader.load_image(d["file_name"])
            anno = d["annotations"]

            metrics, mask_pred = self.evaluate_image(im, anno, crop_size, overlap)

            id = d["image_id"]

            skimage.io.imsave(f"{log_dir}/{id}_mask.png", skimage.img_as_ubyte(mask_pred))

            metrics_all[id] = metrics

            avg_precision += metrics["Precision"]
            avg_recall += metrics["Recall"]
            avg_f1 += metrics["F1-score"]

        avg_precision /= len(metrics_all)
        avg_recall /= len(metrics_all)
        avg_f1 /= len(metrics_all)

        metrics_all["average"] = {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1-score": avg_f1
        }

        with open(f"{log_dir}/metrics.json", "w") as f:
            json.dump(metrics_all, f)

    def evaluate_dataset_cubemap(self, annotation_loader: DatasetLoader, im_dir: str, log_dir: str,
                                 crop_size: Tuple[int, int], overlap: int):
        """
        Evaluate predictions on a dataset.
        Parameters
        ----------
        annotation_loader: DatasetLoader
        Loader used to load annotations (can be different from the one used to load images).
        im_dir: str
        Subdir of the dataset to evaluate on (e.g., 'test', 'val').
        log_dir: str
        Path to logging directory
        crop_size: Tuple[int, int]
        Size of the tiles (in pixels)
        overlap: int
        Amount of overlap between tiles. A warning is printed if
        crop_size is None but overlap is different from 0

        Returns
        -------

        """

        dataset_dicts = annotation_loader.get_window_dicts(im_dir)

        os.makedirs(log_dir, exist_ok=True)

        metrics_all = {}

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0

        faces = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]

        for d in dataset_dicts:
            anno = d["annotations"]
            H, W = d["height"], d["width"]
            mask_true = self.ground_truth_mask(anno, H, W)

            cubemap_pred = {}
            im_id = d["image_id"]
            print(im_id)
            for face in faces:
                # name = d["file_name"]

                face_name = f"{im_id.split('.')[0]}_{face}.png"
                face_im = self.dataset_loader.load_image(face_name, im_dir=im_dir)
                if crop_size is None:
                    if overlap != 0:
                        print("Warning: overlap is set but crop is None, overlap will be ignored.")

                    face_pred = self.predict_image(face_im)
                else:
                    face_pred = self.predict_image_crop(face_im, crop_size, overlap)

                cubemap_pred[face] = face_pred

            mask_pred_reproj = project_cubemap.cubemap2equi(cubemap_pred, (H, W)).astype(bool)

            metrics = self.compute_metrics(mask_true, mask_pred_reproj, verbose=False)

            skimage.io.imsave(f"{log_dir}/{im_id}_mask.png", skimage.img_as_ubyte(mask_pred_reproj))

            metrics_all[im_id] = metrics

            avg_precision += metrics["Precision"]
            avg_recall += metrics["Recall"]
            avg_f1 += metrics["F1-score"]

        avg_precision /= len(metrics_all)
        avg_recall /= len(metrics_all)
        avg_f1 /= len(metrics_all)

        metrics_all["average"] = {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1-score": avg_f1
        }

        with open(f"{log_dir}/metrics.json", "w") as f:
            json.dump(metrics_all, f)
