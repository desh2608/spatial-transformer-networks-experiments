from scipy import misc
import os
from glob import glob
import numpy as np

DIR = "./png/"
DIR_NEW = "./png_small/"

def getArray(filename):
	return misc.imread(filename)

def getDirname(filename):
	if "cheese" in filename:
		return "cheese"
	elif "chocolate" in filename:
		return "chocolate"
	elif "coffee" in filename:
		return "coffee"
	elif "honey" in filename:
		return "honey"
	elif "hotdog" in filename:
		return "hotdog"
	elif "peanut" in filename:
		return "peanut"
	elif "tea" in filename:
		return "tea"

def getLabel(filename):
	if "cheese" in filename:
		return 0
	elif "chocolate" in filename:
		return 1
	elif "coffee" in filename:
		return 2
	elif "honey" in filename:
		return 3
	elif "hotdog" in filename:
		return 4
	elif "peanut" in filename:
		return 5
	elif "tea" in filename:
		return 6
	else:
		print "ERROR"

def resize(filename):
	img_array = getArray(filename)
	img_resized = misc.imresize(img_array,0.2,interp='bilinear',mode=None)
	newfolder = DIR_NEW + getDirname(filename) + "/" 
	newname = newfolder + filename.split('/')[-1]
	if not os.path.exists(newfolder):
		os.makedirs(newfolder)
	misc.imsave(newname,img_resized)


def createSmall():
	if not os.path.exists(DIR_NEW):
		os.makedirs(DIR_NEW)
	result = [y for x in os.walk(DIR) for y in glob(os.path.join(x[0], '*.png'))]
	i = 1
	for f in result:
		if i%10==0:
			print i,"images resized..."
		resize(f)
		i += 1
	print i-1,"images resized total."

def createDataNPZ():
	images=[]
	labels=[]
	result = [y for x in os.walk(DIR_NEW) for y in glob(os.path.join(x[0], '*.png'))]
	for f in result:
		images.append(getArray(f))
		labels.append(getLabel(f))
	images = np.array(images)
	labels = np.array(labels)
	print "Images array shape =",images.shape
	print "Labels array shape =",labels.shape
	np.savez("ego_data",images=images,labels=labels)


# print "Resizing images"
# createSmall()
print "Creating NPZ array"
createDataNPZ()
print "DONE"

