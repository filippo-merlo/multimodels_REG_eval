#%%
import os 

data_dir_path = '/home/fmerlo/data/sceneregstorage'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, '')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'VISIONS_dataset/VISIONS_images')

model_dir = os.path.join(data_dir_path, 'hf_models')
cache_dir = os.path.join(data_dir_path, 'hf_models')
output_dir = os.path.join(data_dir_path, 'attn_eval_output')

temporary_save_dir = os.path.join(data_dir_path, 'temporary_save')

