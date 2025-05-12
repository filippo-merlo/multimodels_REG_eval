#%%
import os 

data_dir_path = '/home/fmerlo/data/sceneregstorage'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'COOCO_dataset/COOCO_data_new.json')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'COOCO_dataset/COOCO_images')

model_dir = '/home/fmerlo/data/sceneregstorage/models/hf_llms_checkpoints'
cache_dir = '/home/fmerlo/data/sceneregstorage/cache_regScene_eval'
output_dir = '/home/fmerlo/data/sceneregstorage/sceneREG_data/output_final'

temporary_save_dir = '/home/fmerlo/data/sceneregstorage/sceneREG_data/temporary_save'