# CS574 CVML Spatial Transformer Networks
Source codes and result data for the Spatial Transformer Networks project for "CS 574 Computer Vision using Machine Learning" course offered in the Fall 2016 semester at IIT Guwahati. (Group-16)

Experiments were performed on the following data sets:

1. Cluttered MNIST handwritten character recognition (cluttered_mnist)
2. Street View House Number recognition (svhn)
3. Georgia Tech Egocentric Activities (gtea)
4. Intel Egocentric Vision (intel)

* Each of the folders contains code, and in some cases, result files for one of the above data sets.
* Data sets have not been provided since they are available online. 
* The files *spatial_transformer.py* and *tf_utils.py* contain code for the STN module.
* Other source files create the model and perform training and testing.
* Note that data source may need to be updated in some cases.

##Dependencies
* Python 2.7
* Numpy
* Scikit-learn
* Tensorflow 0.11 or above

##Notes
The INTEL and GTEA data sets were too large to be trained on my 4GB main memory. So I downsampled the images to 20% size.
For the smaller sampled GTEA data, please look at www.rdesh26.wixsite.com/home/updates/
For the downsampled Intel data and other queries, please raise an issue or contact me at r.desh26@gmail.com