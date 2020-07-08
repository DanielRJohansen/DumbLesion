import random
import Constants
from PIL import Image
import numpy as np
import torch
import time
import Toolbox
import os
import cv2
import sys


class Slicer:
    def __init__(self, sections_lut):
        self.slices_in_section = 7
        self.each_side = self.slices_in_section//2
        self.image_res = 512
        self.work_folder = Constants.work_folder
        self.sections_train = sections_lut[0]
        self.sections_val = sections_lut[1]


        self.buffer = None
        self.batch_ready = False
        if torch.cuda.device_count() > 1:       # This device used to prepare data, not ML
            self.gpu = torch.device('cuda:1')
        else:
            self.gpu = torch.device('cuda:0')
            print("Slicer using main GPU")
        self.cpu = torch.device('cpu')
        self.hu_normalizer = torch.tensor(3000)   # Bone becomes 1, air becomes -0.333

    def getSection(self):
        section = self.__makeRandomSection()
        data, label = self.__randomizeSection(section)
        batch = self.__loadSectionAsTensor(data)
        label = torch.tensor(label, dtype=torch.int8)
        return batch, label


    def __makeRandomSection(self):
        block = random.choice(self.sections_train)
        section = random.choice(block)
        return section

    def __randomizeSection(self, section):
        order = list(range(self.slices_in_section))
        random.shuffle(order)
        randomized_section = []
        for i in range(len(order)):
            randomized_section.append(section[order[i]])

        return randomized_section, order

    def __loadSectionAsTensor(self, data):
        section = [None]*len(data)
        for i, slice_path in enumerate(data):
            slice = torch.load(slice_path)


            section[i] = slice
        section = torch.cat(section, dim=0)
        section = self.__sectionProcessing(section)

        return section



    def __sectionProcessing(self, section):
        section = section.type(dtype=torch.float32)
        section_gpu = section.to(self.gpu)
        hn = self.hu_normalizer.to(self.gpu)
        tensor = torch.div(section_gpu, hn)
        tensor = tensor.to(self.cpu)
        return tensor

    def slowprocess(self, section):
        return np.divide(section, self.hu_normalizer)

        # Prepare data on harddrive




    def makeSlices(self, src_folder):
        folder_names = os.listdir(src_folder)
        for f in folder_names:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(self.work_folder, f)
            if not os.path.exists(dst_path):
                print(dst_path)
                os.mkdir(dst_path)
                images = os.listdir(src_path)
                for im in images:
                    im_path = os.path.join(src_path, im)
                    slice_name = os.path.join(dst_path, im)
                    slice_name = slice_name[:-4]
                    self.makeSlice(im_path, slice_name)


    def makeSlice(self, im_path, slice_name):

        m = np.asarray(cv2.imread(im_path, 0), dtype=np.int8)
        # In extra layer, for easier layering
        if m.shape[1] != 512:
            return False

        m = np.subtract(m, 32768)  # Convert to HU units
        m = np.expand_dims(m, axis=0)
        slice = torch.from_numpy(m)
        slice = slice.type(torch.int16)

        Toolbox.save_tensor(slice, slice_name)
