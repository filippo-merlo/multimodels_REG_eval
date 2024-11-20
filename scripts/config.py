#%%
import os 

data_dir_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'final_dataset_resized.json')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'resized_images')

model_dir = '/mnt/cimec-storage6/shared/hf_lvlms'
cache_dir = '/mnt/cimec-storage6/users/filippo.merlo/cache_regScene_eval'

# model list 
model_list = [
    'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5',
    'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
    'microsoft/kosmos-2-patch14-224',
]



import json

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

i = 0 
for k in dataset.keys():
    print(dataset[k]['target'])
    if dataset[k]['target'] == None:
        print(k)
        i+= 1 
print(i)