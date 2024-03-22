import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

from video_combine import video_combine
from video_split import video_split

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu


# device = torch.device('cpu')

def video_split_2(video_path, output_path):
    # video_path = './video/onepiece_demo.mp4'
    # output_path='./video_output'
    # videos = os.listdir(video_path)
    os.system('ffmpeg -i ' + \
              video_path + \
              ' -f image2 -q:v 2 ' + \
              output_path + \
              '/%8d.png')


def video_combine_2(image_path, video_output_path):
    # image_path = './video_output/%8d.png'
    # video_output_path = './results'
    os.system('ffmpeg -f image2 ' + \
              '-i ' + \
              image_path + \
              video_output_path + \
              '/output.mp4')


test_folder = './video'
output_img_path = 'image_split/'

video_split(test_folder, output_img_path)

v_len=len(os.listdir(output_img_path))

test_folder = output_img_path + '*'
output_img_path = './image_enhance/'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_img_path + str(base) + '.png', output)

video_combine('./image_enhance/','./results',24,(2560,1920),v_len)
