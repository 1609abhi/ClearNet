import numpy as np
import random
from torchvision.transforms import functional as TF

class PairedTransform:
    """
    Paired augmentation matching the reference training pipeline.
    Works on numpy arrays, applies one of 8 augmentations (0=identity).
    """
    def __init__(self, size=256):
        if isinstance(size, (tuple, list)):
            self.size_h, self.size_w = size[0], size[1]
        else:
            self.size_h = self.size_w = size

    def __call__(self, input_img, target_img):
        # PIL → numpy
        input_img  = np.array(input_img)
        target_img = np.array(target_img)

        # Random crop
        H, W = input_img.shape[:2]
        if H < self.size_h or W < self.size_w:
            input_img  = np.array(TF.resize(TF.to_pil_image(input_img),  (self.size_h, self.size_w)))
            target_img = np.array(TF.resize(TF.to_pil_image(target_img), (self.size_h, self.size_w)))
        else:
            ind_H = random.randint(0, H - self.size_h)
            ind_W = random.randint(0, W - self.size_w)
            input_img  = input_img[ind_H:ind_H + self.size_h, ind_W:ind_W + self.size_w]
            target_img = target_img[ind_H:ind_H + self.size_h, ind_W:ind_W + self.size_w]

        # One of 8 augmentations (0 = identity)
        flag_aug   = random.randint(0, 7)
        input_img  = self._augment(input_img,  flag_aug)
        target_img = self._augment(target_img, flag_aug)

        return TF.to_tensor(input_img.copy()), TF.to_tensor(target_img.copy())

    def _augment(self, img, mode):
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))
        else:
            raise Exception('Invalid choice of image transformation')
