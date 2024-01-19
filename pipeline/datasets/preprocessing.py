import torch
import numpy as np

class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, im):  
        assert isinstance(im,np.ndarray)      
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1))
        return torch.from_numpy(im)


class normalization():
    def __init__(self,m,M) -> None:
        self.m_ = m
        self.M_ = M
        
    def __call__(self,im):
        assert isinstance(im,np.ndarray)
        log_im = np.log(np.abs(im)+np.spacing(1))
        num = log_im - self.m_
        den = self.M_-self.m_
        norm = num/den
        norm = np.clip(norm,0,1)
        return (norm).astype(np.float32)