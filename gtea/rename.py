#!/usr/bin/python

import sys
import os
from glob import glob

folder = "./"+ sys.argv[1] + "/"

i = 1

result = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.png'))]

for f in result:
	os.rename(f,folder+str(i)+".png")
	i+=1

os.rmdir(folder+"s1")
os.rmdir(folder+"s2")

print i-1,"files renamed"