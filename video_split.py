import cv2
import os

def video_split(video_path,output_path):
    #video_path = './video'
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        #folder_name = video_path + file_name
        #os.makedirs(folder_name, exist_ok=True)
        vc = cv2.VideoCapture(video_path+'/'+video_name)
        c=0
        rval=vc.isOpened()

        while rval:
            c = c + 1
            rval, frame = vc.read()
            pic_path = output_path
            if rval:
                cv2.imwrite(pic_path + str(c) + '.png', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('spilt_success')
test_folder = './video'
output_img_path = 'image_split/'

video_split(test_folder, output_img_path)