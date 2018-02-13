Final project for ecbm4040 Deep learning for Neural Network
===========================================================
This is a tensorflow implementation of [Multi-digit Number Recognition from Street View Imgery using Deep Convolutional Neural Network](http://arxiv.org/pdf/1312.6802.pdf)

Follow instrctions below will work even on a newly created GCP instance. If you have any problems running the code, please contact us.

Setup
------------
1. Install python dependencies
	```
	$ pip install -r requirements.txt
	```
2. Downloaded [SVHN Dataset] from website or
	```
	$ cd data
	$ ./data_dl.sh
	```


Usage
-------------
	```
	$ python train.py
	```
	If you want to retrain our model, please change the 'pre_trained_model=None' in the main fuction of train.py to pre_trained_model=‘the name of the model’

Accuracy and Loss
-------------------
![Accuracy and Loss](https://bytebucket.org/md3487/ecbm4040_final_project/raw/9ffde6c159cd4b81d6a5d91874332f1e0e3d5420/images/accuracy.png?token=dc66c84fe111f192d8a8a60347ab4c0701d304e2)


