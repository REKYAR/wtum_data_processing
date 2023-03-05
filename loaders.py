import face_recognition
import cv2
import numpy as np
import scipy.io
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil


ctr1 = 0

def load_facial_alge():
    global ctr1
    res={}
    dirpath=".\\facial-age\\archive\\face_age\\face_age"
    for dirname in os.listdir(dirpath):
        f = os.path.join(dirpath, dirname)
        age = int(dirname)
        if age > 80:
            continue
        res[age] = 0
        for filename in os.listdir(f):
            ff = os.path.join(f, filename)
            cv2.imwrite(f"./data_merged/{age}_{ctr1}.jpg", cv2.imread(ff), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            ctr1+=1
            if ctr1 % 1000 == 0:
               print(ctr1)
            res[age] += 1
    return res

def load_utk_face():
    global ctr1
    res={}
    dirpath = '.\\utk-face\\UTKFace'
    for filename in os.listdir(dirpath):
        f = os.path.join(dirpath, filename)
        # checking if it is a file
        if os.path.isfile(f):
            age = int(filename.split('_')[0])
            if age >80:
                continue
            if age in res:
                shutil.copyfile(f,f"./data_merged/{age}_{ctr1}.jpg")
                ctr1+=1
                if ctr1 % 1000 == 0:
                   print(ctr1)
                res[age]+=1
            else:
                res[age]=1
    return res
    

def load_wiki():
    mat = scipy.io.loadmat('./wiki/wiki_crop/wiki.mat')
    #print(mat)
    print(mat['wiki'])
    print(type(mat['wiki']))

    #print(ml)
    #print(mat['wiki'].keys())
#load_wiki()
#load_utk_face()

def summary():
    
    dir1 = load_facial_alge()
    dir2 = load_utk_face()
    for key in dir1:
        if key in dir2:
            dir2[key] += dir1[key]
        else:
            dir2[key] += dir1[key]
    # print(dir2.keys())
    # print(dir2.values())
    #print(sum(dir2.values()))
    #print(len(dir2.keys()))
    plt.bar(dir2.keys(), dir2.values())
    plt.show()

summary()
#load_wiki()