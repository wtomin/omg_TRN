import os
import torch
import torchvision
import torchvision.datasets as datasets

def return_omg(modality, view):
    if modality=='RGB' and view=='body':
        root_data = '/newdisk/omg_TRCNN_code/video_dataset/omg_video_frames'
        train_dict = '/newdisk/omg_TRCNN_code/train_dict.pkl'
        val_dict = '/newdisk/omg_TRCNN_code/val_dict.pkl'
    elif modality =='RGB' and view =='face':
        root_data = '/newdisk/OMGEmotionChallenge/OpenFace_Feature'
        train_dict = '/newdisk/omg_TRCNN_code/train_dict.pkl'
        val_dict = '/newdisk/omg_TRCNN_code/val_dict.pkl'
    elif modality == 'Flow':
        root_data = '/newdisk/omg_TRCNN_code/video_dataset/omg_flow_frames'
        train_dict = '/newdisk/omg_TRCNN_code/train_dict.pkl'
        val_dict = '/newdisk/omg_TRCNN_code/val_dict.pkl' 
    else:
        print('no such modality:'+modality)
        os.exit()
    return root_data, train_dict, val_dict



def return_dataset(dataset, modality, view):
    dict_single = {'omg': return_omg}
    if dataset in dict_single:
        root_data, train_dict, val_dict = dict_single[dataset](modality, view)
    else:
        raise ValueError('Unknown dataset '+dataset)
        
    return root_data, train_dict, val_dict
