import torch
import Toolbox
import numpy as np
import imageio

class Slicer:
    def __init__(self, work_folder):
        self.min_images = 7
        self.each_side = self.min_images//2
        self.image_res = 512
        self.work_folder = work_folder

    def makeSlices(self, src_folder):
        #print("makeSlices called. Are you sure about that? (y/n)")
        #if input() != "y":
        #    return

        for sf in Toolbox.get_subs(src_folder):
            images = Toolbox.get_subs(sf)
            num_ims = len(images)
            if num_ims < self.min_images:
                continue
            i = self.each_side+1
            while i < (num_ims-self.each_side):
                slice = self.makeSlice(images[i-self.each_side-1:i+self.each_side])
                if slice is False:      # Happens when an image is not 512x512
                    continue

                i += 1

    def makeSlice(self, images):
        print(images)
        slice = None
        for im in images:
            piece = torch.from_numpy(np.asarray([imageio.imread(im)], np.int16))
            print(piece.size())
            if piece.size()[1] != 512:
                return False

            if slice is None:
                slice = piece
            else:
                slice = torch.cat((slice, piece),0 )

        print(slice.size())
        Toolbox.save_tensor(slice, self.work_folder)
        exit()
