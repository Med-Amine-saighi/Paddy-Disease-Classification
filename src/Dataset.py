# ====================================================
# Library
# ====================================================
import numpy as np
from PIL import Image
from PIL import ImageFile
# sometimes, you will have images without an ending bit
# this takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import CFG


class classificationDataset:
    def __init__(self,dataframe ,image_paths, targets ,resize=None, augmentations=None):

        self.dataframe = dataframe
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        csv_data = np.array(self.dataframe.iloc[item][['variety','age']].values, dtype=np.float32)
        
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
            
        image = np.array(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

            
        return image, np.array(csv_data) , torch.tensor(targets)

# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose(
        [
           A.Resize(CFG.size, CFG.size),
           A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.Flip(p=0.5),
            
            #A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.5),
            A.ShiftScaleRotate(
                shift_limit = 0.1, scale_limit=0.1, rotate_limit=45, p=0.5
            ),
           
            ToTensorV2(p=1.0),
        ]
    )


    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])