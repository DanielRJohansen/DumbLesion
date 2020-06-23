import Toolbox
import torch
from Slicer import Slicer

source_folder = r"D:\DumbLesion\NIH_scans\Images_png"
work_folder = r"D:\DumbLesion\Slices"


slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
slicer.prepareSliceGetter()
slicer.work()

#slicer.makeSlices(source_folder)