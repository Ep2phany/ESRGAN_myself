import os.path as osp
import os
import glob
import cv2
import numpy as np
import argparse
import torch
from tqdm import tqdm
import RRDBNet_arch as arch
import time

from video_combine import video_combine
from video_split import video_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/RRDB_ESRGAN_x4.pth')
    parser.add_argument('--input', type=str, default='./video')
    parser.add_argument('--split_output', type=str, default='image_split/')
    parser.add_argument('--enhance_output', type=str, default='image_enhance/')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--is_save_enhance', type=bool, default=True)
    parser.add_argument('--is_save_spilt', type=bool, default=True)
    parser.add_argument('--FPS', type=int, default=24)
    parser.add_argument('--size', type=list, default=(2560, 1920))
    args = parser.parse_args()
    print(args)
    # video spilt
    video_split(args.input, args.split_output)
    v_len = len(os.listdir(args.split_output))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model set up
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model.eval()
    model = model.to(device)
    print('Model path {:s}. \nTesting...'.format(args.model_path))

    idx = 0
    with tqdm(total=v_len) as pbar:
        pbar.set_description('Processing Enhance:')
        for path in glob.glob(args.split_output + '*'):
            idx += 1
            base = osp.splitext(osp.basename(path))[0]#获取图像名称
            #print(idx, base)
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
            cv2.imwrite(args.enhance_output + str(base) + '.png', output)#按照获取的名称输出
            time.sleep(0.1)
            pbar.update(1)

    # video combine
    video_combine(args.enhance_output, args.output, args.FPS, args.size, v_len, args.is_save_enhance)


if __name__ == '__main__':
    main()
