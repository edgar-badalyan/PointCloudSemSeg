import json_loader
import custom_transforms
import utils
from PIL import Image

# Detectron2
from detectron2.data import transforms as T

if __name__ == "__main__":
    # To avoid DecompressionBombWarning
    Image.MAX_IMAGE_PIXELS = None

    # Change to fit your dataset path
    DATASET_PATH = "../../../data/Clouds_intensity"

    # You can use this if your dataset has the given format
    # Otherwise, check json_loader.py to see how
    # to implement your own dataset loader
    loader = json_loader.JSONDatasetLoader(DATASET_PATH)

    # define augmentations used in training
    augs = [
        T.FixedSizeCrop(crop_size=(2048, 2048)),
        T.RandomBrightness(0.5, 1.5),
        T.RandomContrast(0.5, 1.5),
    ]

    # If not using default config defined in utils.py,
    # you can pass arguments using this dictionary
    cfg_args = {"OUTPUT_DIR": "output_Clouds_intensity"}

    # Uncomment this line to train
    # train_loop(loader, augs, cfg_args=cfg_args)

    # Uncomment this line to evaluate
    # utils.evaluation(loader, cfg_args, model, log_dir, set="test")
