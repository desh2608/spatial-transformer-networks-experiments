"ego_data.npz" contains the data.

On loading, you will get 2 numpy arrays: 'images' and 'labels'

'images' (3047x81x144x3): 3047 images of size 81x144x3. 
'labels'(3047x1): 3047 labels

NOTE: Original images were 400x720x3 but had to be scaled down to 20% to fit in main memory.

Labels are defined as:
0 - cheese (246 instances)
1 - chocolate (357 instances)
2 - coffee (393 instances)
3 - honey (324 instances)
4 - hotdog (168 instances)
5 - peanut (699 instances)
6 - tea (860 instances)