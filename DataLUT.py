import Toolbox
import csv
import os


def makeOrderLUT(folder, section_size, val_ratio, full_lut=False):
    each_side = section_size//2
    num_sections = 0
    LUT = []
    for sf in Toolbox.get_subs(folder):
        images = Toolbox.get_subs(sf)
        num_ims = len(images)
        if num_ims < section_size:
            continue

        folder_sections = []
        i = each_side
        while i < num_ims-each_side:
            folder_sections.append(images[i-each_side:i+each_side+1])
            i += 1
        LUT.append(folder_sections)

        if not full_lut:    # For testing only
            num_sections += 1
            if num_sections > 100:
                break


    train, val = splitLUT(LUT, val_ratio)
    return [train, val]

def splitLUT(LUT, val_ratio):
    fir_val = int(len(LUT) * (1.-val_ratio))
    return LUT[:fir_val], LUT[fir_val:]




