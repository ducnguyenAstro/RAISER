import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from gettestargs import *
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from sys import exit

args = gettestargs()     #default = 16

# Define parameters
R = int(args.scaling)	#default = 4
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3

trainpath = str(args.folderName)      #default = 'test'

D = int(args.depth)       # default = 16

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

# Read filter from file
filtername = 'filter.p'
if args.filter:
    filtername = args.filter
with open(filtername, "rb") as fp:
    h = pickle.load(fp)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

imagelist.sort()
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    # load image
    origin = cv2.imread(image,-1)
    GREY = is_greyimage(origin)
    if GREY:
        #print("Is GREY")
        grayorigin = origin
    else:
    	# Extract only the luminance in YCbCr
    	ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    	grayorigin = ycrcvorigin[:,:,0]
    
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/(2**D-1), grayorigin.max()/(2**D-1), cv2.NORM_MINMAX)
#    print('After normalization max = {} and min = {}'.format(np.amin(grayorigin), np.amax(grayorigin)))
#    exit()

    # Upscale (bilinear interpolation)
    heightLR, widthLR = grayorigin.shape
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
    heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
    widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)
    upscaledLR = bilinearinterp(widthgridHR, heightgridHR)

    # Calculate predictHR pixels
    heightHR, widthHR = upscaledLR.shape
    predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin))
    operationcount = 0
    totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
    for row in range(margin, heightHR-margin):
        for col in range(margin, widthHR-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                sys.stdout.flush()
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = patch.ravel()
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])

    # Scale back to normal range
    predictHR = np.clip(predictHR.astype('float') * 1.0 * (2**D-1), 0., 1.0 * (2**D-1))
    
    # Bilinear interpolation
    if not GREY:
        # on CbCr field
    	result = np.zeros((heightHR, widthHR, 3))
    	y = ycrcvorigin[:,:,0]
    	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
    	result[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)
    	cr = ycrcvorigin[:,:,1]
    	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    	result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)
    	cv = ycrcvorigin[:,:,2]
    	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')

    	result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)
    	result[margin:heightHR-margin,margin:widthHR-margin,0] = predictHR
    	result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    	result_file = 'results/' + os.path.splitext(os.path.basename(image))[0] + '_R' + str(R)+ '_result.tiff'
    	print( "\rSaving into: ",result_file) 
    	cv2.imwrite(result_file, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    else:   # on grey image
        bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, origin, kind='linear')
        result= bilinearinterp(widthgridHR, heightgridHR)
        result_file = 'results/' + os.path.splitext(os.path.basename(image))[0] + '_R' + str(R)+ '_result.tiff'
        #print('matrix shape:', np.shape(result))
        print( "\rSaving into: ",result_file) 
        # saving image
        try:
            cv2.imwrite(result_file, result.astype(np.uint16))
        except Exception as er:
            print( "Write Error:  ", er)
    imagecount += 1
    # Visualizing the process of RAISR image upscaling
    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(grayorigin, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(upscaledLR, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(predictHR, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(result, interpolation='none')
        plt.show()

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
