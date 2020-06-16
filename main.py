import Toolbox
import torch
from Slicer import Slicer

source_folder = r"D:\CTdeeper\NIH_scans\Images_png"
work_folder = r"E:\DLimages\15_25_60"


slicer = Slicer(work_folder)
slicer.makeSlices(source_folder)