import os
import numpy as np
import torch.utils.data as data

# load mstar image with label
# format : Dir -> target type -> data in npy
class mstar_data(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, paths, process_func):
        
        self.Transform = process_func
        self.path_ = paths

    def __len__(self):
        return len(self.path_)

    def __getitem__(self, item):
        path = self.path_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        label = os.path.basename(os.path.dirname(path))

        im = np.abs(np.load(path))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)
        x = self.Transform(im)
        
        return x,label,name,item

    def get_all_labels(self):
        labels = []
        for p in self.path_:
            label = os.path.basename(os.path.dirname(p))
            labels.append(label)
        return labels