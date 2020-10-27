import os
import shutil

image_dir = '/data/yangxue/dataset/DOTA/val/images/images'
txt_dir = '/data/yangxue/dataset/DOTA/val/labelTxt/labelTxt'

save_image_dir = '/data/yangxue/dataset/OHD-SJTU-LARGE/test/images'
save_txt_dir = '/data/yangxue/dataset/OHD-SJTU-LARGE/test/rotation_txt'


class_list = ['plane', 'small-vehicle', 'large-vehicle', 'ship', 'harbor', 'helicopter']

all_txt = os.listdir(txt_dir)

for t in all_txt:
    fr = open(os.path.join(txt_dir, t), 'r')
    lines = fr.readlines()
    fw = open(os.path.join(save_txt_dir, t), 'w')
    cnt = 0
    for line in lines:
        if len(line.split(' ')) < 9:
            continue

        label = line.split(' ')[8]
        if label not in class_list:
            continue

        box = [int(xy) for xy in line.split(' ')[:8]]

        difficult = line.split(' ')[-1]

        new_line = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(box[0], box[1], box[2], box[3],
                                                                box[4], box[5], box[6], box[7],
                                                                (box[0] + box[2]) // 2,
                                                                (box[1] + box[3]) // 2,
                                                                label, difficult)
        fw.write(new_line)
        cnt += 1
    fw.close()
    fr.close()

    if cnt == 0:
        os.remove(os.path.join(save_txt_dir, t))
    else:
        shutil.copy(os.path.join(image_dir, t.replace('.txt', '.png')), os.path.join(save_image_dir, t.replace('.txt', '.jpg')))