import os
from typing import List, Optional, Tuple

# Personal scripts
from custom_trainer import CustomTrainer
from custom_predictor import Predictor
from evaluator import ModelEvaluator
from dataset_loader import DatasetLoader

# Detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultPredictor


def make_default_config():
    cfg = get_cfg()

    # Change to 'cuda' to use GPU
    cfg.MODEL.DEVICE = 'cpu'

    # Architecture and pre-trained weights
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Name of training dataset
    cfg.DATASETS.TRAIN = ("window_train",)

    # Number of episodes
    cfg.SOLVER.MAX_ITER = 10000

    # Checkpoint period
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    # LR decay
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = [3000, 5000, 6000, 7000, 8000, 9000]

    # Number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Detection threshold is 0.7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1

    # Uncomment next three lines if images are RGBA
    #cfg.INPUT.FORMAT = "RGBA"
    #cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 127.]
    #cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 1.]
    return cfg


def setup_config(loader: DatasetLoader, augs: List = None, cfg_args: dict = None, name: str = "window"):
    """
    Setup a config object.
    Parameters
    ----------
    loader: DatasetLoader
    augs: List
    List of augmentations to be used.
    cfg_args: dict
    Additional arguments for the config.
    name: str
    Name of the  dataset.

    Returns
    -------
    Config.
    """
    # Add other names in the list if needed
    for d in ["train"]:
        DatasetCatalog.register(f"{name}_" + d, lambda d=d: loader.get_window_dicts(d))
        MetadataCatalog.get(f"{name}_" + d).set(things_classes=["window"])
        MetadataCatalog.get(f"{name}_" + d).set(thing_classes=["window"])
    window_metadata = MetadataCatalog.get(f"{name}_train")

    cfg = make_default_config()
    if augs is not None:
        cfg.DATALOADER.AUGS_LIST = augs
    if cfg_args is not None:
        for key, value in cfg_args.items():
            setattr(cfg, key, value)
    return cfg, window_metadata


def train_loop(loader: DatasetLoader, augs: List, cfg_args: Optional[dict] = None,
               last_checkpoint: Optional[str] = None):
    """
    Run train loop
    Parameters
    ----------
    loader: DatasetLoader
    augs: List
    Augmentations to use during training
    cfg_args: dict
    Additional arguments to modify default config.
    last_checkpoint: str
    path to last checkpoint, if any.

    Returns
    -------

    """
    cfg, window_metadata = setup_config(loader, augs=augs, cfg_args=cfg_args)
    if last_checkpoint is not None:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, last_checkpoint)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def setup_evaluator(loader: DatasetLoader, cfg_args: dict, model_name: str, crop: bool) -> ModelEvaluator:
    """
    Create an evaluator that will be used to perform inference and evaluation on datasets.
    Parameters
    ----------
    loader: DatasetLoader
    cfg_args: dict
    Additional arguments to modify default config.
    model_name: str
    Path to model checkpoint file.
    crop: Whether the images are going to be tiled

    Returns
    -------
    ModelEvaluator: evaluator
    """
    cfg, window_metadata = setup_config(loader, cfg_args=cfg_args)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    if not crop and cfg.INPUT.FORMAT != "RGBA":
        predictor = DefaultPredictor(cfg)
    else:
        predictor = Predictor(cfg, crop)

    evaluator = ModelEvaluator(loader, predictor)

    return evaluator


def evaluation(loader: DatasetLoader, cfg_args: dict, model_name: str, log_dir: str,
               set_name: str, crop_size: Optional[Tuple[int, int]] = None, overlap: Optional[int] = 0):
    """
    Evaluate model on given dataset.
    Parameters
    ----------
    loader: DatasetLoader
    cfg_args: dict
    Additional arguments to modify default config.
    model_name: str
    Path to model checkpoint file.
    log_dir: str
    Path to logging directory.
    set_name: str
    Subdir of the dataset to evaluate on (e.g., 'test', 'val').
    crop_size:  Optional[Tuple[int, int]]
    Size of the tiles, if any.
    overlap: Optional[int]
    Amount of overlap between tiles.

    Returns
    -------

    """
    
    evaluator = setup_evaluator(loader, cfg_args, model_name, crop_size is not None)
    
    evaluator.evaluate_dataset(set_name, log_dir, crop_size, overlap)
    
    DatasetCatalog.clear()


def evaluation_cubemap(loader_equirect: DatasetLoader, loader_cubemap: DatasetLoader, cfg_args: dict,
                       model_name: str, log_dir: str, set_name: str,
                       crop_size: Optional[Tuple[int, int]] = None, overlap: Optional[int] = 0):
    """
    Evaluate model on given cubemap dataset.
    Parameters
    ----------
    loader_equirect: DatasetLoader
    Loader for the annotations
    laoder_cubemap: DatasetLoader
    Loader for the images
    cfg_args: dict
    Additional arguments to modify default config.
    model_name: str
    Path to model checkpoint file.
    log_dir: str
    Path to logging directory.
    set_name: str
    Subdir of the dataset to evaluate on (e.g., 'test', 'val').
    crop_size:  Optional[Tuple[int, int]]
    Size of the tiles, if any.
    overlap: Optional[int]
    Amount of overlap between tiles.

    Returns
    -------

    """

    evaluator = setup_evaluator(loader_cubemap, cfg_args, model_name, crop_size is not None)

    evaluator.evaluate_dataset_cubemap(loader_equirect, set_name, log_dir, crop_size, overlap)
    
    DatasetCatalog.clear()
