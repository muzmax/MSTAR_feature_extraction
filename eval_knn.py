import os

import pipeline.models.ViT as vits
from pipeline.utils import load_model
from pipeline.predictor.mstar_knn import mstar_knn
from pipeline.logger import setup_logger,LOGGER

import torch.nn as nn
from torchvision import transforms 

class no_deep(nn.Module):
    def __init__(self,resize_dim=128):
        super().__init__()
        self.rs_dim = resize_dim
        self.rs = transforms.Resize((resize_dim,resize_dim),antialias=True)
        # self.rs = transforms.CenterCrop((resize_dim,resize_dim))
    def forward(self, x):
        b,c,_,_ = x.shape
        out = self.rs(x)
        return out.contiguous().view(1,self.rs_dim**2)

def eval_knn(params):
     # ============ building network and dataset pipeline ... ============
    setup_logger(out_file=params['logger_path'],setup_msg=False)
    
    if params['arch'] in vits.__dict__.keys():
            model = vits.__dict__[params['arch']](patch_size=params['patch_size'])
            model.set_patch_drop(params['patch_drop'])
            model.to(params['device'])
            load_model(model, params['model_path'])
            model.eval()
    elif params['arch'] == None:
         model = no_deep()
         model.to(params['device'])
         model.eval()
    else :
        print("Unknow architecture: {}".format(params['arch']))
    mstar_eval = mstar_knn(params['normalization'],
                           params['eval_path'],
                           params['labeled_path'],
                           pca=params['pca'],
                           pca_dim=params['pca_dim'],
                           device=params['device'])
    # ============ mstar evaluation ... ============
    results = mstar_eval(model, nb_knn=params['nb_knn'], temperature=params['temperature'])
    LOGGER.info('Results : {}'.format(results))


if __name__ == '__main__':
    params = {}
    # Knn parameters
    params['nb_knn'] = [1,2,3,4,5,6,7,8]
    params['temperature'] = 0.07
    # Dataset parameters
    params['normalization'] = [-9.,4.] # normalization after the log transform 
    params['eval_path'] = './data/eval'
    params['labeled_path'] = './data/labeled'
    # Network parameters
    params['device'] = 'cuda'
    params['model_path'] = 'pipeline/weights/sethi_t_8' # vit_t/8
    params['arch'] = None # None for a simple unroling of the image or 'vit_tiny' for the trained network
    params['patch_size'] = 8
    params['patch_drop'] = 0
    params['pca'] = True # Do a pca on top of the extracted features
    params['pca_dim'] = 50 # number of components of the pca
    # Path and name of the logger
    params['logger_path'] = './logger/knn_mstar_pca_10_label.txt'


    eval_knn(params)
    
    

   
