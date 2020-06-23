import torch
import Toolbox
import numpy as np
import imageio
import os
import random

class Slicer:
    def __init__(self, work_folder):
        self.slices_in_section = 7
        self.each_side = self.slices_in_section//2
        self.image_res = 512
        self.work_folder = work_folder
        self.centerslices = [] # [[folder_path, 3, 4, 5]]  or [path, useable middles]
        self.num_centers = 0
        self.min_hu = -1000.
        self.max_hu = 3000.
        self.n_val_per_hu = 2. / (self.max_hu - self.min_hu)

    def prepareSliceGetter(self):
        for sf in Toolbox.get_subs(self.work_folder):
            centers = []
            i = self.each_side
            while i < len(Toolbox.get_subs(sf))-self.each_side:
                centers.append(i)
                i += 1
            self.centerslices.append([sf, centers])
            break


    def makeRandomSection(self):
        block = random.choice(self.centerslices)
        center = random.choice(block[1])
        section = []
        images = Toolbox.get_subs(block[0])
        for i in range(self.slices_in_section):
            slice = images[center - self.each_side + i]
            section.append(slice)
        return section

    def randomizeSection(self, section):
        order = list(range(self.slices_in_section))
        random.shuffle(order)
        randomized_section = []
        for i in range(len(order)):
            randomized_section.append(section[order[i]])
        reverse_order = order.copy()
        reverse_order.reverse()

        return randomized_section, [order, reverse_order]

    def work(self):
        while True:
            section = self.makeRandomSection()
            data, label = self.randomizeSection(section)







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
                    try:
                        slice = self.makeSlice(im_path, normalize=True)
                        if slice is not False:
                            Toolbox.save_tensor(slice, slice_name)
                    except:
                        print(slice_name, "failed")

    def makeSlice(self, im_path, normalize):
        m = np.asarray([imageio.imread(im_path)], np.int16)   # In extra layer, for easier layering
        m = np.subtract(m, 32768)
        slice = torch.from_numpy(m)

        if slice.size()[1] != 512:
            return False

        if normalize:
            slice = self.normalizeSlice(slice)
        return slice

    def normalizeSlice(self, slice):
        for i in range(self.image_res):
            for j in range(self.image_res):
                n_val = -1 + slice[0][i][j] * self.n_val_per_hu
                if n_val < -1: n_val = -1
                elif n_val > 1: n_val = 1
                slice[0][i][j] = n_val
        return slice

