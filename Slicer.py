import random
import threading
class Slicer:
    def __init__(self, work_folder, sections_lut):
        self.slices_in_section = 7
        self.each_side = self.slices_in_section//2
        self.image_res = 512
        self.work_folder = work_folder
        self.sections_train = sections_lut[0]
        self.sections_val = sections_lut[1]

        self.buffer = None
        self.batch_ready = False

    def getSection(self):
        section = self.__makeRandomSection()
        data, label = self.__randomizeSection(section)
        return [data, label]


    def __makeRandomSection(self):
        block = random.choice(self.sections_train)
        section = random.choice(block)
        return section

    def __randomizeSection(self, section):
        order = list(range(self.slices_in_section))
        random.shuffle(order)
        randomized_section = []
        for i in range(len(order)):
            randomized_section.append(section[order[i]])

        return randomized_section, order








