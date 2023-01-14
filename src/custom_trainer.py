from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader

class CustomTrainer(DefaultTrainer):
    """
    Trainer class based on the DefaultTrainer of detectron2
    """
        
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        dataloader = build_detection_train_loader(cfg,
                                                  mapper=DatasetMapper(cfg, is_train=True, augmentations=cfg.DATALOADER.AUGS_LIST))
        return dataloader
