import Toolbox
import torch
import CSVReader
import DataLUT
from Batcher import Batcher



if __name__ == '__main__':
    source_folder = r"D:\DumbLesion\NIH_scans\Images_png"
    work_folder = r"F:\DumbLesion\NIH_images"
    LUT_path = r"F:\DumbLesion\LUT.txt"
    LUT = DataLUT.makeOrderLUT(work_folder, LUT_path, section_size=7, val_ratio=0.2, full_lut=False)

    batcher = Batcher(work_folder, LUT, num_agents=4, batch_size=32)
    while True:
        batcher.getBatch()
        print("batch")












#slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
#slicer.prepareSliceGetter()
#slicer.work()

#slicer.makeSlices(source_folder)