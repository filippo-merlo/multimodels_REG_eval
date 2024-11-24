#%%
import os 

data_dir_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'final_dataset_resized.json')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'resized_images')

model_dir = '/mnt/cimec-storage6/shared/hf_lvlms'
cache_dir = '/mnt/cimec-storage6/users/filippo.merlo/cache_regScene_eval'
