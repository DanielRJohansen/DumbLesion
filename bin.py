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

"""