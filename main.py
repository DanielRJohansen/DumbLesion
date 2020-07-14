import Toolbox
import torch
import CSVReader
import DataLUT
from BatcherV2 import Batcher
import Constants
import time
from DumbLesion import DumbLesionNet

if __name__ == '__main__':
    dst = r"F:\DumbLesion\AreasOfConfidence\Train"

    #Toolbox.highest_lowest(r"F:\DumbLesion\Slices\000001_01_01\109.pt")
    #Toolbox.highest_lowest(r"F:\DumbLesion\Slices\000004_03_02\136.pt")
    #Toolbox.printJpegs()
    Toolbox.makeAOC(r"D:\DumbLesion\NIH_scans\Images_png",r"F:\DumbLesion\AreasOfConfidence\Train", r"D:\DumbLesion\DL_info.csv", im_size=256)
    #Toolbox.makeZLabels(r"F:\DumbLesion\AreasOfConfidence\Train", r"D:\DumbLesion\DL_info.csv")
    #Toolbox.makeAOCLabels(Constants.work_folder, Constants.INFO, num_areas=32)
    #Toolbox.visualizeLabel(im_path=r"D:\DumbLesion\NIH_scans\Images_png\000865_06_01\114.png",
    #                       label_path=r"F:\DumbLesion\AreasOfConfidence\Train\000865_06_01\114_AOCLabel.pt")
    #exit()
    #LUT = DataLUT.makeOrderLUT(Constants.work_folder, section_size=Constants.section_depth,
    #                           val_ratio=0.2, full_lut=False)

    #NN = DumbLesionNet(batcher=Batcher(Constants.work_folder, label_type="z"))
    #NN._train()














#slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
#slicer.prepareSliceGetter()
#slicer.work()

#slicer.makeSlices(source_folder)