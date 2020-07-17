from glob import glob
import os
import torch
import cv2
import numpy as np
import csv

def get_subs(folder):
    return glob(folder + r"\*")

def min_images(folder):

    fewest = 9999
    bins = [0]*1000
    for sf in get_subs(folder):
        num_images = len(get_subs(sf))
        bins[num_images] += 1
        if num_images < fewest:
            fewest = num_images
            print(sf, fewest)

    print("Fewest images in all folders:", fewest)
    print(bins)

def delete_files(folder, f_format=".csv", suffix_len=4):
    print("Removing {} files from {}".format(f_format, folder))
    for sf in get_subs(folder):
        for file in get_subs(sf):
            if file[-suffix_len:] == f_format:
                os.remove(file)


def save_tensor(tensor, file_name):
    torch.save(tensor, file_name     + ".pt")

def highest_lowest(path):
    im = torch.load(path)
    print(path)
    print("Lowest: ", torch.min(im))
    print("Highest ", torch.max(im))

def getFolderAndCenter(string):
    i = len(string)-1
    while True:
        if string[i] == '_':
            break
        i -= 1
    folder = string[:i]
    center = string[i+1:]
    return folder, center

def makeSectionList(folder, center):
    ims = get_subs(folder)
    i = 0
    while i < len(ims):
        if ims[i] == center:
            break
        i += 1
    section = ims[i-3:i+4]
    return section

from matplotlib import pyplot as plt

def loadSection(section_paths, im_size):
    section = []
    for path in section_paths:
        im = cv2.imread(path, cv2.CV_16U)
        im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
        section.append(np.asarray(im))

    section = torch.tensor(section, dtype=torch.float32)
    section = section.type(torch.float32)
    section = torch.sub(section, 32768)
    im = section.numpy()
    hist, bins = np.histogram(im.ravel(), 400, [-1000, 3000])   # Use hist
    return section, hist

def saveSectionAndHist(section, hist, folder, name):
    name = name[:-4]
    name = os.path.join(folder, name)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    hist = torch.tensor(hist, dtype=torch.float32)

    torch.save(section, name + "_section.pt")
    torch.save(hist, name + "_hist.pt")


def makeAOC(src, dst, data_file, im_size):
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            sf, center = getFolderAndCenter(row[0])
            print(sf)

            sf_path = os.path.join(src, sf)
            center_path = os.path.join(src, sf, center)
            dst_path = os.path.join(dst, sf)

            section = makeSectionList(sf_path, center_path)
            if len(section) != 7:

                continue
            section, hist = loadSection(section, im_size=im_size)
            saveSectionAndHist(section, hist, dst_path, center)
import ast
def toList(string):
    list = ast.literal_eval(string)
    return list

def drawBB(im_path, AOC_label):
    pass

def getRelativeBoxPosition(row, num_areas):
    box = toList(row[6])
    size = toList(row[13])[0]
    box = np.multiply(box, num_areas/size)
    box = np.round(box)
    return box

def makeAOCLabel(box, num_areas):
    label = np.zeros((num_areas, num_areas), dtype=np.float32)
    for y in range(num_areas):
        for x in range(num_areas):
            if y >= box[1] and y <= box[3] and x >= box[0] and x <= box[2]:
                label[y][x] = 1
    return torch.tensor(label)

def makeAOCLabels(dst, data_file, num_areas):
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            sf, center = getFolderAndCenter(row[0])
            print(sf)
            if elementExists(dst, sf, center[:-4]):
                box = getRelativeBoxPosition(row, num_areas)
                label = makeAOCLabel(box, num_areas)
                name = os.path.join(dst, sf, center[:-4])
                torch.save(label, name + "_AOCLabel.pt")


def cleanElement(string):
    index = 0
    while index < len(string):
        if string[index] == '_':
            break
        index += 1
    return string[:index]

def elementExists(f, sf, name):
    if not os.path.isdir(os.path.join(f, sf)):
        return False
    elemsinfolder = os.listdir(os.path.join(f, sf))
    elemsinfolder = list(set(list(map(cleanElement, elemsinfolder))))  # Stackoverflow, obviously..
    for e in elemsinfolder:
        if name == e:
            return True
    return False


def makeZLabels(dst, data_file):
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            sf, center = getFolderAndCenter(row[0])
            print(sf)
            name = center[:-4]
            path = os.path.join(dst, sf, name)
            if elementExists(dst, sf, name):
                z = torch.tensor(toList(row[8])[2], dtype=torch.float32)
                torch.save(z, path + "_zlabel.pt")

def visualizeLabel(im_path, label_path):
    label = torch.load(label_path)
    im = cv2.imread(im_path, cv2.CV_16U)
    print(im.shape)
    label = torch.add(label, 1)
    label = torch.div(label, 2).numpy()
    #label = label.astype(np.uint16)
    label = cv2.resize(label, (512, 512))

    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    im = cv2.multiply(im, label)
    cv2.imshow("result", im)
    cv2.waitKey()










def printJpegs():   #No need, all is png :)
    for sf in get_subs(r"D:\DumbLesion\NIH_scans\Images_png"):
        for im in get_subs(sf):
            if im[-3:] != "png":
                print(im)