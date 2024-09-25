import cv2
import numpy as np
import torch
from typing import Tuple

def load_image_for_model(image_path:str) -> Tuple[torch.Tensor, np.ndarray]:
    '''
    Loads an image from path
    
    Args:
        image_path: Image path
    
    Returns:
        img: Torch tensor with shape [B, C, H, W]
        img_org: numpy array with shape [H, W, C]
    '''
    img_org = cv2.imread(image_path)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = np.transpose(img_org, (2,0,1))/255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img, img_org