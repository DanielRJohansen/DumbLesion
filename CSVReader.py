import csv
import os
import ast

def getLUTs(LUT_path):


    trainLUT = []
    valLUT = []

    with open(LUT_path) as file:
        string = file.read()

        LUT = ast.literal_eval(string)
        print(LUT)
    file.close()



