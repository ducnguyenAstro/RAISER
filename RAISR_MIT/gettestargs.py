import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", help="Use file as filter")
    parser.add_argument("-p", "--plot", help="Visualizing the process of RAISR image upscaling", action="store_true")
    parser.add_argument("-d", "--depth", help="Depth of image (default = 16)", default=16)
    parser.add_argument("-R", "--scaling", help="Scaling number (default = 4)", default=4)
    parser.add_argument("-fn", "--folderName", help="Folder Name of test set (default = \'test\')", default='test')
    args = parser.parse_args()
    return args

def is_greyimage(image):
    if image.ndim == 2:
        return True # this image is grayscale
    elif image.ndim == 3:
        return False # image is either RGB or YCbCr colorspace
