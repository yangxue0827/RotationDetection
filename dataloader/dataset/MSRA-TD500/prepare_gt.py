import os
import sys
import glob
import math
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm

def rotate(theta, x, y):
    rotatex = math.cos(theta) * x - math.sin(theta) * y
    rotatey = math.cos(theta) * y + math.sin(theta) * x
    return rotatex, rotatey
 
def xy_rorate(theta, x, y, cx, cy):
    r_x, r_y = rotate(theta, x - cx, y - cy)
    return cx + r_x, cy + r_y

def rec_rotate(x, y, w, h, t):
    cx = x + w / 2
    cy = y + h / 2
    x1, y1 = xy_rorate(t, x, y, cx, cy)
    x2, y2 = xy_rorate(t, x + w, y, cx, cy)
    x3, y3 = xy_rorate(t, x, y + h, cx, cy)
    x4, y4 = xy_rorate(t, x + w, y + h, cx, cy)
    return x1, y1,  x3, y3, x4, y4,x2, y2
 

def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


def convert_label(gt_path, dst_path):
    f = open(gt_path,'r')
    savestr = ''
    for line in f:
        _, diffcult, lx, ly, w, h, theta  = [eval(x) for x in line.strip().split(' ')]
        quads = rec_rotate(lx, ly, w, h, theta)
        char_quads = ','.join([str(round(x)) for x in sort_corners(np.array(quads)[np.newaxis, :]).squeeze()])
        text = ',text\n' if diffcult == 0 else ',###\n'
        char_quads += text
        savestr += char_quads
    savef = open(dst_path,'w')
    savef.write(savestr)
    savef.close()


def generate_txt_labels(src_dir, eval_dir):
    label_paths = glob.glob(os.path.join(src_dir, '*.gt'))
    label_txt_path = osp.join(eval_dir, 'temp')
    if os.path.exists(label_txt_path):
        shutil.rmtree(label_txt_path)
    os.mkdir(label_txt_path)

    pbar = tqdm(label_paths)
    for label in pbar:
        im_name = os.path.split(label)[1].strip('.gt').strip('IMG_')
        pbar.set_description("gt.zip of MSRA_TD500 is generated in {}".format(eval_dir))
        label_txt = osp.join(label_txt_path, 'gt_img_' + im_name + '.txt')
        convert_label(label, label_txt)
    gt_zip = os.path.join(eval_dir, 'gt.zip')
    os.system('zip -j {} {}'.format(gt_zip, label_txt_path + '/*')) 
    shutil.rmtree(label_txt_path)

if __name__ == '__main__':
    root_dir='/data-input/RotationDet'
    generate_txt_labels(src_dir=os.path.join(root_dir, 'data/MSRA_TD500/test'),
                        eval_dir=os.path.join(root_dir,'DOTA_devkit/MSRA_TD500/eval')
                        )
    print('done!')
