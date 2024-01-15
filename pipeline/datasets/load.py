import glob
import random
import os
import numpy as np

def load_separated_paths(dir1,dir2,batch_sz=1):
    train_paths = []
    eval_paths = []
    sub_dirs_1 = glob.glob('{}/*'.format(dir1))
    sub_dirs_2 = glob.glob('{}/*'.format(dir2))
    classes_1 = [class_name for class_name in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, class_name))]
    classes_2 = [class_name for class_name in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, class_name))]
    assert classes_1 == classes_2

    total_im = 0
    for sub_dir in sub_dirs_1:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        nb_im = len(filepaths)
        total_im += nb_im
        train_paths = [*train_paths,*filepaths]
    print("{} - Found {} dirs with a total of {} training images and {} steps".format(dir1,len(sub_dirs_1),total_im,int(total_im//batch_sz)))

    total_im = 0
    for sub_dir in sub_dirs_2:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        nb_im = len(filepaths)
        total_im += nb_im
        eval_paths = [*eval_paths,*filepaths]
    print("{} - Found {} dirs with a total of {} evaluation images ".format(dir2,len(sub_dirs_2),total_im))

    return train_paths,eval_paths,classes_1