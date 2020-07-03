import Toolbox
import torch
import CSVReader
import DataLUT
from Batcher import Batcher
import Constants
import time
from DumbLesion import DumbLesionNet

if __name__ == '__main__':




    LUT = DataLUT.makeOrderLUT(Constants.work_folder, section_size=Constants.section_size,
                               val_ratio=0.2, full_lut=False)

    NN = DumbLesionNet(batcher=Batcher(LUT))
    NN.go()














#slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
#slicer.prepareSliceGetter()
#slicer.work()

#slicer.makeSlices(source_folder)