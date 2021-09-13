import os


filter_thres = 0.25
version = 'RetinaNet_MLT_BCD_1x_20210818'
detector = 'bcd'
res_files = os.listdir('../../../tools/{}/test_mlt/{}/mlt_res'.format(detector, version))
filter_res_path = '../../../tools/{}/test_mlt/{}_{}/mlt_res'.format(detector, version, filter_thres)

if not os.path.exists('../../../tools/{}/test_mlt/{}_{}/mlt_res'.format(detector, version, filter_thres)):
    os.makedirs('../../../tools/{}/test_mlt/{}_{}/mlt_res'.format(detector, version, filter_thres))

for rf in res_files:
    fr = open('../../../tools/{}/test_mlt/{}/mlt_res/{}'.format(detector, version, rf), 'r')
    fw = open('../../../tools/{}/test_mlt/{}_{}/mlt_res/{}'.format(detector, version, filter_thres, rf), 'w')
    lines = fr.readlines()
    for line in lines:
        if float(line.split(',')[-1].split('\n')[0]) > filter_thres:
            fw.write(line)
    fr.close()
    fw.close()
