import cv2
import os
#ffmpeg -i .\video\onepiece_demo.mp4 -f image2 -q:v 2  ./video_output/%3d.png
#ffmpeg -f image2 -i .\video_output\%8d.png out.mp4

def video_test():
    video_path = './video/onepiece_demo.mp4'
    output_path='./video_output'
    #videos = os.listdir(video_path)
    os.system('ffmpeg -i ' + \
              video_path + \
              ' -f image2 -q:v 2 ' + \
              output_path + \
              '/%8d.png'
              )
def video_combine():
    image_path = './image_enhance/%08d.png'
    video_output_path = ' ./results'
    os.system('ffmpeg -f image2 ' + \
                '-i ' + \
              image_path + \
              video_output_path + \
              '/output.mp4' )
video_combine()


