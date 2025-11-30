import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class ScratchSegDataset(Dataset):
    def __init__(self, names_list, img_dir="/home/personal/Desktop/mowito/data/images", mask_dir="/home/personal/Desktop/mowito/data/masks",
                 augment=False, img_size=512):
        self.names = names_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.img_size = img_size

        self.t_train = A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=8, p=0.4),
            A.MotionBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

        self.t_val = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, name)
        img = cv2.imread(img_path)[:,:,::-1]

        mask_path = os.path.join(self.mask_dir, name)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype("uint8")
        else:
            mask = np.zeros(img.shape[:2], dtype="uint8")

        if self.augment:
            aug = self.t_train(image=img, mask=mask)
        else:
            aug = self.t_val(image=img, mask=mask)

        img_t = aug["image"]
        mask_t = aug["mask"].unsqueeze(0).float()

        return img_t, mask_t
