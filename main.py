import Toolbox
import torch
import CSVReader
import DataLUT
from Batcher import Batcher
import Constants
import time
import sys
from Slicer import Slicer
if __name__ == '__main__':




    LUT = DataLUT.makeOrderLUT(Constants.work_folder, Constants.LUT_path, section_size=Constants.section_size,
                               val_ratio=0.2, full_lut=False)

    batcher = Batcher(LUT)
    t0 = time.time()
    batches = []
    while True:
        batch = batcher.getBatch()
        batches.append(batch)
        print("Avg. batch time:", (time.time()-t0)/len(batches))
        print("Num batches: ", len(batches))













#slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
#slicer.prepareSliceGetter()
#slicer.work()

#slicer.makeSlices(source_folder)