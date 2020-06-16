from glob import glob
import os
import torch

def get_subs(folder):
    return glob(folder + r"\*")

def min_images(folder):

    fewest = 9999
    bins = [0]*1000
    for sf in get_subs(folder):
        num_images = len(get_subs(sf))
        bins[num_images] += 1
        if num_images < fewest:
            fewest = num_images
            print(sf, fewest)

    print("Fewest images in all folders:", fewest)
    print(bins)

def delete_files(folder, f_format=".csv", suffix_len=4):
    print("Removing {} files from {}".format(f_format, folder))
    for sf in get_subs(folder):
        for file in get_subs(sf):
            if file[-suffix_len:] == f_format:
                os.remove(file)


def save_tensor(tensor, dst_folder):
    torch.save(tensor, dst_folder + r"\file" + ".pt")
