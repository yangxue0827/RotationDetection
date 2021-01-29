import cv2
import os.path as osp
import os
import argparse


def argsparser():
    parser = argparse.ArgumentParser("Transform images to videos")
    # General Setting
    parser.add_argument('--input_dir', help='input file dir', type=str, default='')
    parser.add_argument('--output_dir', help='output file dir', type=str, default='')
    parser.add_argument('--fps', help='fps', type=int, default=60)
    parser.add_argument('--width', help='width', type=int, default=2880)   # [3840, 2880]
    parser.add_argument('--height', help='height', type=int, default=1620)  # [2160 1620]

    return parser.parse_args()

if __name__ == '__main__':

    forcc = cv2.VideoWriter_fourcc(*'mp4v') 

    args = argsparser()
    input_dir = args.input_dir
    output_dir = args.output_dir

    fps = args.fps
    size = (args.width, args.height)

    for filename in os.listdir(input_dir):

        video_name = filename+'.mp4'     
        save_path = output_dir + filename

        if not osp.exists(save_path):
            os.makedirs(save_path)

        videoWriter = cv2.VideoWriter(osp.join(save_path, video_name), forcc, fps, size)

        imgs = os.listdir(osp.join(input_dir, filename))
        imgs.sort(key=lambda x: int(x.split('.')[0]))

        for img in imgs:
            assert img is not None
            img_id = int(img.split('.')[0])
            if img_id in range(0, 843) or img_id in range(3453, 4631) or img_id in range(5751, 7174):
                continue
            img_path = osp.join(osp.join(input_dir, filename), img)
            print(img_path)
            read_img = cv2.imread(img_path)[:args.height, :args.width, :]
            # assert read_img is not None
            # assert read_img.shape[0] == size[1]
            # assert read_img.shape[1] == size[0]
            videoWriter.write(read_img)
            try:
                os.system("cls")
            except:
                os.system("clear")

        videoWriter.release()

