import math
import random
import numpy as np
import matplotlib.pyplot as plt
import alphashape

from libs.utils.mask_sample import points_sampling


def carttopolar(x, y, x0=0, y0=0):
    '''
    cartisian to polar coordinate system with origin shift to x0,y0
    '''
    x1 = x - x0
    y1 = y - y0
    # print('(x0,y0)sort',x0,y0)
    r = np.sqrt(x1 ** 2 + y1 ** 2)
    t = np.arctan2(y1, x1) * 180 / math.pi
    if y1 < 0:
        t += 360
    # print('x,y,r,t',x,y,r,t)
    return r, t


def sort_aniclkwise(xy_list,x0=None,y0=None):
    '''
    Sort points anit clockwise with x0 y0 as origin
    '''
    if x0 is None and y0 is None:
        (x0, y0) = np.mean(xy_list, axis=0).tolist()
    elif x0 is None:
        (x0, _) = np.mean(xy_list, axis=0).tolist()
    elif y0 is None:
        (_, y0) = np.mean(xy_list, axis=0).tolist()
    # print('origin used:',[x0,y0])

    for i in range(len(xy_list)):
          xy_list[i].append(i)

    xy_list1 = sorted(xy_list, key=lambda a_entry: carttopolar(a_entry[0], a_entry[1],x0,y0)[1])

    sort_index = []  
    for x in xy_list1:
        sort_index.append(x[2])
        del x[2]

    return np.array(xy_list1, np.float32)


def sort_aniclkwise_batch(polys):
    return np.array([sort_aniclkwise(poly.tolist()) for poly in polys], np.float32)


def draw(pts):
    color_dict = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    idx = random.randint(0,7)
    pts = np.array(pts).reshape(-1,2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    plt.scatter(xs,ys)
    plt.plot(xs, ys)


def concave_hull(points, point_num, alpha=0.95):
    points = np.array(points)
    alpha *= alphashape.optimizealpha(points)
    hull = alphashape.alphashape(points, alpha)
    hull_pts = hull.exterior.coords.xy
    hull_pts = np.array(hull_pts)
    hull_pts = np.concatenate([np.reshape(hull_pts[0][:-1], [-1, 1]), np.reshape(hull_pts[1][:-1], [-1, 1])], axis=1)
    hull_pts = np.reshape(hull_pts, [-1, 2])
    hull_pts = points_sampling(hull_pts, point_num)
    return hull_pts


def concave_hull_batch(polys, point_num, alpha):
    return np.array([concave_hull(poly, point_num, alpha) for poly in polys], np.float32)


if __name__ == '__main__':
    pts = [[1, 1], [0, 2], [2, 1], [2, 2], [2, 0], [0, 1], [1, 0], [0, 0]]
    points = np.array([(17, 158), (15, 135), (38, 183), (43, 19), (93, 88), (96, 140), (149, 163), (128, 248), (216, 265),
              (248, 210), (223, 167), (256, 151), (331, 214), (340, 187), (316, 53), (298, 35), (182, 0), (121, 42)])

    # draw(points)
    # spts = sort_aniclkwise(pts)
    spts = concave_hull(points)
    print(spts)
    draw(points)
    plt.show()

    # import ipdb; ipdb.set_trace()
