"""

 # Prepare data on harddrive

    def makeSlices(self, src_folder):
        folder_names = os.listdir(src_folder)
        for f in folder_names:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(self.work_folder, f)
            if not os.path.exists(dst_path):
                print(dst_path)
                os.mkdir(dst_path)
                images = os.listdir(src_path)
                for im in images:
                    im_path = os.path.join(src_path, im)
                    slice_name = os.path.join(dst_path, im)
                    slice_name = slice_name[:-4]
                    try:
                        self.makeSlice(im_path, slice_name)
                    except:
                        print(slice_name, "Failed")


    def makeSlice(self, im_path, slice_name):
        m = np.asarray([imageio.imread(im_path)], np.int16)   # In extra layer, for easier layering
        if m.shape[1] != 512:
            return False

        m = np.subtract(m, 32768)   # Convert to HU units

        slice = torch.from_numpy(m)
        Toolbox.save_tensor(slice, slice_name)


"""