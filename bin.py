"""

section = []
        for slice_path in data:
            slice = Image.open(slice_path)
            slice = np.asarray(slice, dtype=np.float32)
            slice = np.expand_dims(slice, axis=0)
            section.append(slice)
        section = np.concatenate(section, axis=0)
        section = self.__tensorrize(section)
        section = self.__sectionProcessing(section)


    def __tensorrize(self, section):
        return torch.tensor(section)











    calc dims
           batch = torch.zeros((1,16,515,515))
        batch = self.Branch1_conv(batch)
        print(batch.shape)
        print(1*16*128*128)
        print(int(np.prod(batch.size())))
"""