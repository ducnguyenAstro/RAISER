import argparse
from time import strftime, gmtime
from os import system

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qmatrix", help="Use file \"q.p\"as Q matrix", action="store_true")
    parser.add_argument("-v", "--vmatrix", help="Use file \"v.p\"as V matrix", action="store_true")
    parser.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
    parser.add_argument("-d", "--depth", help="Depth of image (default = 16)", default=16)
    parser.add_argument("-R", "--scaling", help="Scaling number (default = 4)", default=4)
    parser.add_argument("-fn", "--folderName", help="Folder Name of test set (default = \'train\')", default='train')
    parser.add_argument("-z", "--zip", help="Zip filters aftermath", action="store_true")
    args = parser.parse_args()
    return args

def is_greyimage(image):
    if image.ndim == 2:
        return True # this image is grayscale
    elif image.ndim == 3:
        return False # image is either RGB or YCbCr colorspace

def zipFilter(path,R,f1,f2):
    #path is train folder name
    #f1, f2: filter file name
    StoreFileName = "trainingFilters_" + path+ '_R' + R + "_" + strftime("%j%H%M%S", gmtime())
    cmd = 'zip ' +  StoreFileName + '.zip' + ' '+ f1 + ' '+ f2
    system(cmd)

