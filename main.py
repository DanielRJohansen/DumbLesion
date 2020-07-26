import Toolbox
import torch
import CSVReader
import DataLUT
from BatcherV2 import Batcher
import Constants
import time
from DumbLesion import DumbLesionNet




def demonstration():
    model_path = r"./Models/8_-0.12_model.pt"

    NN = DumbLesionNet(output_type="AoC", num_val_ims=Constants.num_val_ims, CNN_trainable=False)
    NN.loadModel(model_path=model_path)
    NN.visualValidation()


def train():
    dst = r"F:\DumbLesion\AreasOfConfidence\Train"
    model_path = r"./Models/z_run/299_0.949_basemodel.pt"

    NN = DumbLesionNet(output_type="AoC", num_val_ims=Constants.num_val_ims, CNN_trainable=False)
    NN.base.loadModel(model_path=model_path)
    NN._train(best_acc=-999, save_base_only=False)


if __name__ == '__main__':

    #train()
    demonstration()







    #print("Model's state_dict:")
    #for param_tensor in NN.base.state_dict():
    #    print(param_tensor, "\t", NN.base.state_dict()[param_tensor].size())

    #NN.base.saveModel(Constants.model_path)

# Toolbox.highest_lowest(r"F:\DumbLesion\Slices\000001_01_01\109.pt")
# Toolbox.highest_lowest(r"F:\DumbLesion\Slices\000004_03_02\136.pt")
# Toolbox.printJpegs()
# Toolbox.makeAOC(r"D:\DumbLesion\NIH_scans\Images_png",r"F:\DumbLesion\AreasOfConfidence\Train", r"D:\DumbLesion\DL_info.csv", im_size=256)
# Toolbox.makeZLabels(r"F:\DumbLesion\AreasOfConfidence\Train", r"D:\DumbLesion\DL_info.csv")
# Toolbox.makeAOCLabels(r"F:\DumbLesion\AreasOfConfidence\Train", Constants.INFO, num_areas=32)
# Toolbox.visualizeLabel(im_path=r"D:\DumbLesion\NIH_scans\Images_png\000865_06_01\114.png",
#                       label_path=r"F:\DumbLesion\AreasOfConfidence\Train\000865_06_01\114_AOCLabel.pt")
# exit()
# LUT = DataLUT.makeOrderLUT(Constants.work_folder, section_size=Constants.section_depth,
#








#slicer = Slicer(work_folder)
#slicer.makeSlices(source_folder)
#slicer.prepareSliceGetter()
#slicer.work()

#slicer.makeSlices(source_folder)