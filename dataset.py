import torch.utils.data as data
import pdb 
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pickle

class VideoRecord(object):
    def __init__(self, row, video, utter, root_path, label_name, phase):
        self._data = row
        self.video = video
        self.utterance = utter
        self._label_name = label_name
        self._phase= phase
        if not label_name in row.keys():
            print('Wrong label name')
            os.exit()
        self.root_path = root_path
    @property
    def path(self):
        if len(os.listdir(self.root_path))>3:
            return os.path.join(self.root_path, self.video, self.utterance)
        else:
            return os.path.join(self.root_path, self._phase, self.video, 'processed', self.utterance+'_aligned')
        

    @property
    def num_frames(self):
        return int(self._data['num_frames'])
    @property
    def list_faces(self):
        return list(self._data['list_faces'])
    @property
    def label(self):
        return self._data[self._label_name]


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, dict_file , label_name,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, phase='Train'):

        self.root_path = root_path # root path is where the video frames are stored
        self.dict_file = dict_file  # train dict or val dict
        self.label_name = label_name # one of 'arousal', 'valence' and 'EmotionMaxVote'
        self.num_segments = num_segments # number of segments in TRNMultiscale Module
        self.new_length = new_length # new length is? 
        self.modality = modality # flow or RGB, or RGBDiff. but in TRN I think only RGB is used
        self.image_tmpl = image_tmpl #image naming strategy, used to find iamge name
        self.transform = transform # transform function
        self.random_shift = random_shift 
        self.test_mode = test_mode
        self.phase = phase
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_dict()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')] #open image 
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')] #error opening image, only load the first frame?
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx-1)*5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_dict(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        data_dict = pickle.load(open(self.dict_file,'rb'))
        self.utterance_list = list()
        for video in data_dict.keys():
            for utter in data_dict[video].keys():
                item = data_dict[video][utter]
                vr = VideoRecord(item, video, utter, self.root_path, label_name=self.label_name, phase=self.phase)
                if len(vr.list_faces)>=self.num_segments:
                    self.utterance_list.append(vr)
        print('video number: %d'%(len(data_dict.keys())))
        print('utterance number:%d'%(len(self.utterance_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (len(record.list_faces) - self.new_length + 1) // self.num_segments # for example, the input num of frames is 30, num_segments is 8, average_duration is 3
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) # results randomly uniformly sampled 8 frames
        elif record.num_frames > self.num_segments: # if num_frames> num_segment
            offsets = np.sort(randint(len(record.list_faces) - self.new_length + 1, size=self.num_segments)) # sample randomly, 8 frames
        else:
            offsets = [0]* self.num_segments
        return [record.list_faces[idx] for idx in offsets]

    def _get_val_indices(self, record):
        if len(record.list_faces) > self.num_segments + self.new_length - 1:
            tick = (len(record.list_faces) - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) # uniformly sample? No it returns index larger than num_frames, which doesn't make sense
        else:
            offsets = [0]* self.num_segments# if validation video does not have enough frames >8, return zeros?
        return [record.list_faces[idx] for idx in offsets]

    def _get_test_indices(self, record):

        tick = (len(record.list_faces) - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return [record.list_faces[idx] for idx in offsets]

    def __getitem__(self, index):
        record = self.utterance_list[index] 
        # check this is a legit video folder
        while not os.path.exists(os.path.join(record.path, self.image_tmpl.format(1))):
            print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))) # if the first frame doesn't exist, print out the first frame of this video
            index = np.random.randint(len(self.utterance_list))
            record = self.utterance_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list() 
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images) # transformation changes 8, 224, 244 to 24, 224, 224,

        return process_data, record.label

    def __len__(self):
        return len(self.utterance_list)

if __name__ == "__main__":
    import datasets_video
    from transforms import *
    import torchvision
    root_path, train_dict, val_dict = datasets_video.return_dataset('omg', 'RGB','face')
    label_name = 'arousal'
    num_segments = 8 
     
    dataset = TSNDataSet(root_path, train_dict, label_name, num_segments=num_segments, phase='Train',
                   new_length=1,
                   modality='RGB',
                   image_tmpl='frame_det_00_{:06d}.bmp',
                   transform=torchvision.transforms.Compose([
                                    GroupScale(256),
                                    GroupRandomCrop(224),
                                    Stack(True),
                                    ToTorchFormatTensor(),
                                    GroupNormalize(
                                        mean=[.485, .456, .406],
                                        std=[.229, .224, .225]),
                                        IdentityTransform()
                   ]))

    sample = dataset[45]
    sample = dataset[19]