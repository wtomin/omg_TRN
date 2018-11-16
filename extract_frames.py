import os
import threading
import glob

NUM_THREADS = 50
VIDEO_ROOT = '/newdisk/OMGEmotionChallenge/Videos'         # Downloaded webm videos
FRAME_ROOT = '/newdisk/omg_TRCNN_code/video_frames'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    utter_name = video.split('/')[-1].split('.')[0]
    video_name = video.split('/')[-2]
    os.system('ffmpeg -i {} -vf scale=256:256 {}/{}/{}/{}'.format(video, FRAME_ROOT, video_name, utter_name, tmpl))


def target(video_list):
    for video in video_list:
        utter_name = video.split('/')[-1].split('.')[0]
        video_name = video.split('/')[-2]
        save_dir = os.path.join(FRAME_ROOT, video_name, utter_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = glob.glob(os.path.join(VIDEO_ROOT, 'Train', '*'))
video_list.remove(os.path.join(VIDEO_ROOT, 'Train', 'youtube_videos_temp'))
video_list += glob.glob(os.path.join(VIDEO_ROOT, 'Validation', '*'))
video_list.remove(os.path.join(VIDEO_ROOT, 'Validation', 'youtube_videos_temp'))
video_list += glob.glob(os.path.join(VIDEO_ROOT, 'Test','*'))
video_list.remove(os.path.join(VIDEO_ROOT, 'Test', 'youtube_videos_temp'))

video_list = [video for video in video_list if os.path.isdir(video) ] # make sure it's a directory
utter_list = list()
for video_dir in video_list:
    utter_path = glob.glob(os.path.join(video_dir, '*.mp4'))
    utter_list.extend(utter_path)
splits = list(split(utter_list, NUM_THREADS))

threads = []
for i, split in enumerate(splits):
    print("Process: {}/{}".format(i+1, len(splits)))
    thread = threading.Thread(target=target, args=(split,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
