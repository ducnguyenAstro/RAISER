import numpy as np
import math
import sys, os
from PIL import Image
import glob
import time
import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--Predict", help="Predict Images Folder")
    parser.add_argument("-CV", "--CV", help="Cross Validation folder")
    args = parser.parse_args()
    return args
    
# calculate linear cost error of 2 images
def linearCostError( file1, file2):
    # return: error and True, -1 and False for if images are differnt size
    img1 = Image.open(file1)
    arr1 = np.array(img1)
    img2 = Image.open(file2)
    arr2 = np.array(img2)
    
    try:
        print (file1, 'and ', file2)
        return np.square(np.subtract(arr1, arr2)).mean() * 0.5, True
    except Exception as ex:
        print (' are NOT the same size, Error = ', ex)
        return -1, False

# estimate average linear cost error of mulitples files on 2 different folder, one is predict and the other is cross validation 
def errorEstimate(predictFolder, CVFolder):
    # build predict file name list
    predictList = []
    for parent, dirnames, filenames in os.walk(predictFolder):
        for filename in filenames:
            if filename.lower().endswith(('.tif','.tiff')):
                predictList.append(os.path.join(parent, filename))
    predictList.sort()

    # build CV file name list
    CVList = []
    for parent, dirnames, filenames in os.walk(CVFolder):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                CVList.append(os.path.join(parent, filename))
    CVList.sort()
    
    # find the matching files
    count = 0; errorSum = 0; sizeCount = 0;
    for predict in predictList:
        for CVfile in CVList:
            if os.path.split(predict)[-1].startswith(os.path.split(CVfile)[-1][:-4]):
                count +=1; 
                error, matchingBool = linearCostError(predict, CVfile)
                if matchingBool:
                    errorSum += error; 
                    sizeCount +=1;
                     
        
    print(" There are ", count, " matching files")
    print(" And ", sizeCount, " images matching size")
         
    try:
        return (errorSum/sizeCount)                
    except:
        print("No matching files found")
        return -1
        
def main():
    args = gettestargs()
    t0 = time.time()
    logfile = args.Predict + '/Cost_Error_' + time.strftime("%j%H%M%S", time.gmtime()) + '.txt'    
    f = open(logfile, 'a')
    default_stdout = sys.stdout
    sys.stdout = f
    costError = errorEstimate(args.Predict, args.CV)
    if costError != -1:
        print ('\r Average linear cost error is :', costError)              
    print (' Elapsed time : ', (time.time()- t0) /60,' mins')
    f.close()
    sys.stdout = default_stdout
    
    print('\rCost Error = ', costError, '\r Done in ', (time.time()- t0) /60,' mins')
    print('See the log at ', logfile)
    
if __name__ == "__main__":
    main()

