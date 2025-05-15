#%%
import os 

data_dir_path = '/scratch/merlo003'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'COOCO_dataset/COOCO_data_new.json')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'COOCO_dataset/COOCO_images')

model_dir = os.path.join(data_dir_path, 'hf_models')
cache_dir = os.path.join(data_dir_path, 'hf_models')
output_dir = os.path.join(data_dir_path, 'eval_output')

temporary_save_dir = os.path.join(data_dir_path, 'temporary_save')