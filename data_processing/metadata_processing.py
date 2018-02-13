import numpy as np
import _pickle as pickle
import h5py

'''
This function parses the .mat metadata information and transform the metadata into python dict datas structure.
@param filepath 	the filepath of .mat metadatafile
@return metadata	the output python dict type metadata
'''

def get_metadata(filepath):
    f = h5py.File(filepath)

    metadata= {}
    metadata['height'] = []
    metadata['label'] = []
    metadata['left'] = []
    metadata['top'] = []
    metadata['width'] = []
    
    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(int(obj[0][0]))
        else:
            for k in range(obj.shape[0]):
                vals.append(int(f[obj[k][0]][0][0]))
        metadata[name].append(vals)
    
    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)
        
    return metadata

'''
This function convert the original vision of metadata into multiple objects data vision. Not used in the training process.
@param data		original dict data
@return metadata_dict 	converted data
'''
def alter(data):
    metadata_dict = {}
    for i in range(0, len(data['label'])):
        metadata_dict[str(i + 1)] = {}
        metadata_dict[str(i + 1)]['label'] = data['label'][i]
        metadata_dict[str(i + 1)]['top'] = data['label'][i]
        metadata_dict[str(i + 1)]['left'] = data['label'][i]
        metadata_dict[str(i + 1)]['width'] = data['label'][i]
        metadata_dict[str(i + 1)]['height'] = data['label'][i]
    
    return metadata_dict
'''
This function extends the original label data, which has a length less than 7, to fixed length 7 labels for training usage.
@param data 	dict type metadata
@return data 	label extended metadata
'''
def extend_label(data):
    for i in range(0, len(data['label'])):
        data['label'][i].insert(0, len(data['label'][i]))
        while len(data['label'][i]) <= 6:
            data['label'][i].append(10)
    
    return data
'''
This function get the border of all the bounding boxes for one image in metadata and merges all boxes into one which contains all digits in a image. The merged bounding box has 30% extended range for sampling random shifting training images.
@param metadata 	metadata dict
@return ret 		merged bounding boxes

'''
def get_digit_border(metadata):
    ret = []
    for i in range(0, len(metadata['label'])):
        top = max(0, min(metadata['top'][i]))
        left = max(0, min(metadata['left'][i]))
        bot = 0
        right = 0
        for idx in range(len(metadata['top'][i])):
            bot = max(bot, metadata['height'][i][idx] + metadata['top'][i][idx])
            right = max(right, metadata['left'][i][idx] + metadata['width'][i][idx])
        
        ret.append([top, bot, left, right])
    
    return ret
        
            
            
    
