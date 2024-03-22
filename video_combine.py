import os

import cv2

def video_combine(data_path,output_path,fps,size,video_len):
    #data_path = "./image_enhance/"
    #fps = 24  # 视频帧率
    #size = (640,480)  # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter(output_path+'/output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    for i in range(video_len):
        image_path = data_path+str(i+1)+'.png'
        print(image_path)
        img = cv2.imread(image_path)
        video.write(img)
        os.remove(image_path)

    video.release()
    cv2.destroyAllWindows()
video_combine('./image_split/', './results',60,(640,480),181)