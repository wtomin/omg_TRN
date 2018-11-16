# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
import pandas as pd
import numpy as np
import pickle
dataset_name = 'video_frames' 

train_dict = 'train_dict.pkl'
val_dict = 'val_dict.pkl'



def make_dict_label(train_csv, dict_file_path = None, partition_col=None, vd_name_col =None, utr_name_col = None, label_name_list = [], \
                     is_partitioned= True, is_utterance=False, openface_results_root = None):
    """ Make a dictionary with [partition]:[sample_name]:[label_1], [label_2], [num_frames], [list_faces]...
    Args:
        train_csv: a csv file which specifys the video name (utterance name if applicable), labels, parititions
        dict_file_path: a file path for saving samples and labels
        partition_col: column name of partition, e.g., train, dev, val
        vd_name_col: column of video name
        label_name_list: a list of names of labels, e.g., emotion, age
        is_partitioned: whether the dataset has been partitioned already
        is_utterance: whether the video is subdivided into utterances
        frames_root: where the frames are stored
        openface_results: the directory of openface results, about detecting face results
    """
    data = pd.read_csv(train_csv, skipinitialspace=True, sep="\s+|;|:|,",engine="python")
    dictionary = {}
    if is_partitioned:
        p_df = data[partition_col]
        partitions = list(np.unique(p_df))
        for partition in partitions:
            dictionary[partition] = {}
            part_df = data[data[partition_col]==partition]
            if is_utterance:
                for index, row in part_df.iterrows():
                    vd_name = row[vd_name_col]
                    utr_name = row[utr_name_col].split('.')[0]
                    if vd_name not in dictionary[partition].keys():
                        dictionary[partition][vd_name] = {}
                    dictionary[partition][vd_name][utr_name] = {}
                    for label_name in label_name_list:
                        dictionary[partition][vd_name][utr_name][label_name] = row[label_name]
                    # record number of frames, and the frames list that contains faces
                    opface_csv = os.path.join(openface_results_root, vd_name, 'processed', utr_name+'.csv')
                    if not os.path.exists(opface_csv):
                        dictionary[partition][vd_name].pop(utr_name,None)
                    else:
                        opface_df = pd.read_csv(opface_csv, skipinitialspace=True, sep="\s+|;|:|,",engine="python")
                        faces_only_df = opface_df[opface_df['success']==1]
                        faces_list = list(faces_only_df['frame'])
                        dictionary[partition][vd_name][utr_name]['num_frames'] = len(faces_only_df)
                        dictionary[partition][vd_name][utr_name]['list_faces'] = faces_list
            else:
                #if no utterance exists
                for index, row in part_df.iterrows():
                    vd_name = row[vd_name_col]
                    dictionary[partition][vd_name] = {}
                    for label_name in label_name_list:
                        dictionary[partition][vd_name][label_name] = row[label_name]
                    # record number of frames, and the frames list that contains faces
                    opface_csv = os.path.join(openface_results_root, vd_name, 'processed', vd_name+'.csv')
                    if not os.path.exists(opface_csv):
                        dictionary[partition][vd_name].pop(vd_name,None)
                    else:
                        opface_df = pd.read_csv(opface_csv, skipinitialspace=True, sep="\s+|;|:|,",engine="python")
                        faces_only_df = opface_df[opface_df['success']==1]
                        faces_list = list(faces_only_df['frame'])
                        dictionary[partition][vd_name][utr_name]['num_frames'] = len(faces_only_df)
                        dictionary[partition][vd_name][utr_name]['list_faces'] = faces_list
    else:
        #if orginal video dataset has not been paritioned
        if is_utterance:
            for index, row in data.iterrows():
                vd_name = row[vd_name_col]
                utr_name = row[utr_name_col].split('.')[0]
                if vd_name not in dictionary.keys():
                    dictionary[vd_name] = {}
                dictionary[vd_name][utr_name] = {}
                for label_name in label_name_list:
                    dictionary[vd_name][utr_name][label_name] = row[label_name]
                # record number of frames, and the frames list that contains faces
                opface_csv = os.path.join(openface_results_root, vd_name, 'processed', utr_name+'.csv')
                if not os.path.exists(opface_csv):
                    dictionary[vd_name].pop(utr_name,None)
                else:
                    opface_df = pd.read_csv(opface_csv, skipinitialspace=True, sep="\s+|;|:|,",engine="python")
                    faces_only_df = opface_df[opface_df['success']==1]
                    faces_list = list(faces_only_df['frame'])
                    dictionary[vd_name][utr_name]['num_frames'] = len(faces_only_df)
                    dictionary[vd_name][utr_name]['list_faces'] = faces_list
        else:
            #if no utterance exists
            for index, row in data.iterrows():
                vd_name = row[vd_name_col]
                dictionary[vd_name] = {}
                for label_name in label_name_list:
                    dictionary[vd_name][label_name] = row[label_name]
                # record number of frames, and the frames list that contains faces
                opface_csv = os.path.join(openface_results_root, vd_name, 'processed', vd_name+'.csv')
                if not os.path.exists(opface_csv):
                    dictionary[vd_name].pop(vd_name,None)
                else:
                    opface_df = pd.read_csv(opface_csv, skipinitialspace=True, sep="\s+|;|:|,",engine="python")
                    faces_only_df = opface_df[opface_df['success']==1]
                    faces_list = list(faces_only_df['frame'])
                    dictionary[vd_name][utr_name]['num_frames'] = len(faces_only_df)
                    dictionary[vd_name][utr_name]['list_faces'] = faces_list
    pickle.dump(dictionary, open(dict_file_path, 'wb'))
    print("Dictionary saved in :" + dict_file_path)
    return dictionary

# generate label dictionary
train_csv = '/newdisk/OMGEmotionChallenge/new_omg_TrainVideos.csv'
if not os.path.exists('train_dict.pkl'):
    train_dict = make_dict_label(train_csv, dict_file_path ='train_dict.pkl', is_partitioned = False, vd_name_col='video', \
                utr_name_col = 'utterance', label_name_list=['arousal', 'valence','EmotionMaxVote'], is_utterance=True, openface_results_root='/newdisk/OMGEmotionChallenge/OpenFace_Feature/Train')

val_csv = '/newdisk/OMGEmotionChallenge/new_omg_ValidationVideos.csv'
if not os.path.exists('val_dict.pkl'):
    val_dict = make_dict_label(val_csv, dict_file_path='val_dict.pkl', is_partitioned = False, vd_name_col='video', 
                           utr_name_col = 'utterance', label_name_list=['arousal', 'valence','EmotionMaxVote'], is_utterance=True, openface_results_root='/newdisk/OMGEmotionChallenge/OpenFace_Feature/Validation')