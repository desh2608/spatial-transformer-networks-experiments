from scipy import misc
import os
from glob import glob
import numpy as np
import sys

DIR = "./"+ sys.argv[1] + "/"
DIR_NEW = DIR[:-1] + "_small/"

def resize(filename):
	img_array = misc.imread(filename)
	img_resized = misc.imresize(img_array,0.2,interp='bilinear',mode=None)
	newname = DIR_NEW + filename.split('/')[-1]
	misc.imsave(newname,img_resized)


def createSmall():
	if not os.path.exists(DIR_NEW):
		os.makedirs(DIR_NEW)
	result = [y for x in os.walk(DIR) for y in glob(os.path.join(x[0], '*.jpg'))]
	i = 0
	for f in result:
		resize(f)
		i += 1
		if i%100==0:
			print i,"images resized..."
	print i-1,"images resized total."

# def createDataNPZ():
# 	images=[]
# 	labels=[]
# 	result = [y for x in os.walk(DIR_NEW) for y in glob(os.path.join(x[0], '*.jpg'))]
# 	for f in result:
# 		images.append(getArray(f))
# 		labels.append(getLabel(f))
# 	images = np.array(images)
# 	labels = np.array(labels)
# 	print "Images array shape =",images.shape
# 	print "Labels array shape =",labels.shape
# 	np.savez("ego_data",images=images,labels=labels)


print "Resizing images"
createSmall()
# print "Creating NPZ array"
# createDataNPZ()
print "DONE"

