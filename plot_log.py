from __future__ import print_function, absolute_import

import re
import os.path as osp
import pdb
import argparse
import pdb
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--logtxt', default='train.txt', help='path to datatxt')

opt = parser.parse_args()

def writetxt(lines, path):
    fobj = open(path, 'a')
    for line in lines:
        fobj.write(str(line) + '\n')
    fobj.close()

def main(fileroad):
    if not osp.exists(fileroad):
        print("Can not find file : {file}".format(file = fileroad))
        return
    print('open file')
    with open(fileroad, 'r') as f:
        Loss_pacs = []
        Loss_pacs_temp = []
        loss_pacs_str = re.compile('accuracy of the photo dataset:')
        
        Loss_search = re.compile(r'([\d]+\.[\d]+)')
        for line in f:
            map_Loss = Loss_search.findall(line)
            pacs_str = loss_pacs_str.findall(line)
            if map_Loss:
                if "photo" in line:
                    Loss_pacs.append(float(map_Loss[0]))
        for k in range(0,int(len(Loss_pacs)/2-1)):
            Loss_pacs_temp.append(Loss_pacs[k*2])
            Loss_pacs_m_temp.append(Loss_pacs[k*2+1])
        
    return Loss_pacs_temp, Loss_pacs_m_temp
    #return Loss_pacs, Loss_pacs_m
Loss_pacs, Loss_pacs_temp = main(opt.logtxt)
Loss_pacs = Loss_pacs[101:]
Loss_pacs_temp = Loss_pacs_temp[101:]

pacs_max = max(Loss_pacs)
test_max = Loss_pacs_temp[Loss_pacs.index(pacs_max)]
print('photo_max',pacs_max,test_max)
pacs_max = max(Loss_pacs_temp)
print('photo_test_max',pacs_max)
