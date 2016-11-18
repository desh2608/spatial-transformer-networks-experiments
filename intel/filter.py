from scipy import misc
import scipy.io as sio
import os
from glob import glob
import numpy as np
from shutil import copyfile
import sys

DIR = "./Data/"
DIR_NEW = DIR[:-1] + "_filtered/"

def filterFiles(old_folder,new_folder):
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)
		print "Created folder",new_folder
	result = [y for x in os.walk(old_folder) for y in glob(os.path.join(x[0], '*.jpg'))]
	copied = 0
	filtered = 0
	l = sio.loadmat('./labels/'+sys.argv[1]+'.mat')['L']
	labels = []
	for i,f in enumerate(result):
		if l[i][0] != 0:
			copyfile(f,new_folder+f.split('/')[-1])
			labels.append(l[i][0])
			copied += 1
			if copied%10==0:
				print ("%d (%.2f%%) images copied"%(copied,float(copied*100)/(copied+filtered)))
		else:
			filtered += 1
	print ("%d (%.2f%%) images copied"%(copied,float(copied*100)/(copied+filtered)))
	labels = np.array(labels)
	np.save('./labels/'+sys.argv[1],labels)
	print ("New labels saved in file /labels/"+sys.argv[1]+".npy")


print "Filtering files from folder",DIR
filterFiles(DIR,DIR_NEW)
print "DONE"

