import numpy as np
import data_processing.metadata_processing as mp
import tensorflow as tf
import imageio  
from skimage import transform 
from matplotlib import pyplot as plt
from random import randint
import os
import json



class dataset:
    def __init__(self):
        ##############################
        #class initilazation function#
        ##############################
        self.train_image_path = "./data/train/"
        self.test_image_path = "./data/test/"
        self.idx_train = 0	#train dataset index pointer for batch
        self.idx_test = 0	#test dataset index pointer for batch
        self.rand_idx_train = np.random.permutation(33402) #shuffle index pemutation
        self.rand_idx_test = np.random.permutation(13068)  #shuffle index permutation
    '''
    #This function is used for data batch(test and train)#
    #@param data original 64 * 64 size datasset, can be train or test dataset
    #@param label original 7 size label set, can be train or test label set
    #@param batch_size the size of batch this function return
    #@param is_train indicates whether the data is train dataset or test dataset
    #@param shuffle controls whether the order of data returned is shuffled or not, default is Flase
    #@return a batch of images and labels with given batch size
    '''
    def build_batch(self, data, labels, batch_size, is_train, shuffle=False):
        rand_idx_train = self.rand_idx_train
        rand_idx_test = self.rand_idx_test
        
        if shuffle is False:
            if is_train is True:
                idx = self.idx_train
                if idx + batch_size > 33402:
                    batch = np.concatenate((data[idx:33402], data[0:idx + batch_size - 33402]), axis = 0)
                    label = np.concatenate((labels[idx:33402], labels[0:idx + batch_size - 33402]), axis = 0)
                    self.idx_train = idx + batch_size - 33402
                else:
                    batch = data[idx:idx + batch_size]
                    label = labels[idx:idx + batch_size]
                    self.idx_train = idx + batch_size
            else:
                idx = self.idx_test
                if idx + batch_size > 13068:
                    batch = np.concatenate((data[idx:13068], data[0:idx + batch_size - 13068]), axis = 0)
                    label = np.concatenate((labels[idx:13068], labels[0:idx + batch_size - 13068]), axis = 0)
                    self.idx_test = idx + batch_size - 13068
                else:
                    batch = data[idx:idx + batch_size]
                    label = labels[idx:idx + batch_size]
                    self.idx_test = idx + batch_size
        else:
            if is_train is True:
                idx = self.idx_train
                if idx + batch_size > 33402:
                    first_part = rand_idx_train[idx:33402]
                    rand_idx_train = np.random.permutation(33402)
                    second_part = rand_idx_train[0:idx + batch_size - 33402]
                    self.rand_idx_train = rand_idx_train
                    mask = np.concatenate((first_part, second_part), axis = 0)
                    
                    batch = data[mask]
                    label = labels[mask]
                    self.idx_train = idx + batch_size - 33402
                else:
                    batch = data[rand_idx_train[idx:idx + batch_size]]
                    label = labels[rand_idx_train[idx:idx + batch_size]]
                    self.idx_train = idx + batch_size
            else:
                idx = self.idx_test
                if idx + batch_size > 13068:
                    first_part = rand_idx_test[idx:13068]
                    rand_idx_test = np.random.permutation(13068)
                    second_part = rand_idx_test[0:idx + batch_size - 13068]
                    self.rand_idx_test = rand_idx_test
                    mask = np.concatenate((first_part, second_part), axis = 0)
                    
                    batch = data[mask]
                    label = labels[mask]
                    self.idx_test = idx + batch_size - 13068
                else:
                    batch = data[rand_idx_test[idx:idx + batch_size]]
                    label = labels[rand_idx_test[idx:idx + batch_size]]
                    self.idx_test = idx + batch_size

        '''
        crop a random shifted 54 * 54 size image out of the original 64 * 64 size image as stated in the papaer
        '''
        cropped = np.zeros((batch_size, 54, 54, 3))
        if is_train is True:
            for i in range(batch_size):
                rand = np.random.randint(10, size=2)
                cropped[i] = batch[i, rand[0]:rand[0]+54, rand[1]:rand[1] + 54, :]    
        else:
            cropped = batch[:, 5:59, 5:59, :]
        return cropped, label
                
                               
    '''
    #This function loads the JSON format metadat information(or output the metadata information files if they are not existed) and train/test dataset in given path.
    @param size the reshape size of image stored in memory
    @return train_image 33402*64*64*3 numpy array of all train images
    @return train_label 33402*7 numpyarray of all train labels
    @return test_image 13068*64*64*3 numpy array of all test images
    @return test_label 13068*7 numpy array of all test labels
    '''
    def load_image(self, size):
        train_image_path = self.train_image_path
        test_image_path = self.test_image_path
        train_data = []
        test_data = []
        print ("Generating metadata for training.")
        if not os.path.exists('./data_processing/metadata_train.json'):
            metadata = mp.get_metadata('./data/train/digitStruct.mat')
            with open('./data_processing/metadata_train.json', 'w') as outfile:
                json.dump(metadata, outfile, indent=2)
        else:
            metadata = json.load(open('./data_processing/metadata_train.json'))

        meta_train = mp.get_digit_border(metadata)
        print ("Done")
        print ("Load training dataset labels")
        metadata = mp.extend_label(metadata)
        train_labels = np.stack(metadata['label'])
        print ("Done")
        print ()

        print ("Generating metadata for testing.")
        if not os.path.exists('./data_processing/metadata_test.json'):
            metadata = mp.get_metadata('./data/test/digitStruct.mat')
            with open('./data_processing/metadata_test.json', 'w') as outfile:
                json.dump(metadata, outfile, indent=2)
        else:
            metadata = json.load(open('./data_processing/metadata_test.json'))
        meta_test = mp.get_digit_border(metadata)
        print ("Done")
        print ("Load test dataset labels")
        metadata = mp.extend_label(metadata)
        test_labels = np.stack(metadata['label'])
        print ("Done")
        print ()
        print ("loading traning data:")
        for i in range(1, 33403):
            if i % 1000 is 0 or i == 33402:
                print (str(i) + "/" + str(33402))
            try:
                image = imageio.imread(train_image_path + str(i) + ".png")
                chop_image = image[meta_train[i - 1][0]:meta_train[i - 1][1], meta_train[i - 1][2]:meta_train[i - 1][3]]
                '''
                plt.imshow(image, interpolation='nearest')
                plt.show()
                '''
                resized_image = transform.resize(chop_image, size)
                x = randint(1, 10)
                y = randint(1, 10)
                train_data.append(resized_image)
                #train_data.append(resized_image[y:y + 54, x:x + 54])
            except:
                print (i)
                print (image.shape)
                print (chop_image.shape)
                print (resized_image.shape)
                print ()

        print ("loading traning data:")      
        for i in range(1, 13069):
            if i % 1000 is 0 or i == 13068:
                print (str(i) + "/" + str(13068))
            try:
                #Load the image using skimage package
                image = imageio.imread(test_image_path + str(i) + ".png")
                chop_image = image[meta_test[i - 1][0]:meta_test[i - 1][1], meta_test[i - 1][2]:meta_test[i - 1][3]]
                '''
                plt.imshow(image, interpolation='nearest')
                plt.show()
                '''
                resized_image = transform.resize(chop_image, size)
                x = randint(1, 10)
                y = randint(1, 10)

                #train_data.append(resized_image[y:y + 54, x:x + 54])
                train_data.append(resized_image)

                test_data.append(resized_image)
            except:
                print (i)
                print (image.shape)
                print (chop_image.shape)
                print (resized_image.shape)
                print ()



        train_data = np.stack(train_data, axis=0)
        test_data = np.stack(test_data, axis=0)
        return train_data, test_data, train_labels, test_labels

    def load_labels(self):
        if not os.path.exists('./data_processing/metadata_train.json'):
            metadata = mp.get_metadata('./data/train/digitStruct.mat')
            with open('./data_processing/metadata_train.json', 'w') as outfile:
                json.dump(metadata, outfile, indent=2)
        else:
            metadata = json.load(open('./data_processing/metadata_train.json'))


        metadata = mp.extend_label(metadata)
        #print (metadata['label'])
        train_labels = np.stack(metadata['label'])    

        print ("Generating metadata for testing.")
        if not os.path.exists('./data_processing/metadata_test.json'):
            metadata = mp.get_metadata('./data/test/digitStruct.mat')
            with open('./data_processing/metadata_test.json', 'w') as outfile:
                json.dump(metadata, outfile, indent=2)
        else:
            metadata = json.load(open('./data_processing/metadata_test.json'))
        metadata = mp.extend_label(metadata)
        test_labels = np.stack(metadata['label'])

        return train_labels, test_labels
