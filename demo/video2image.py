""" 从视频读取帧保存为图片"""
import cv2
import os.path as osp
import os

input_dir = './videos/'
output_dir = './images/'

for filename in os.listdir(input_dir):
    output_path = output_dir+filename

    if '5542' in filename:
        continue

    if not osp.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(input_dir+filename)
    c = 0
    rval=cap.isOpened()
    while rval:
        # get a frame
        rval, frame = cap.read()
        if rval:
            cv2.imwrite(output_path+'/'+str(c) + '.jpg',frame) 
            c += 1
            print(c)
            cv2.waitKey(1)
        else:
            break
    cap.release()
