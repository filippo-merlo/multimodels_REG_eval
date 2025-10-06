#%%
import os 

data_dir_path = '/home/fmerlo/data/sceneregstorage'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'VISIONS_dataset/S1_boundingbox_critical_object.csv')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'VISIONS_dataset/VISIONS_images')

model_dir = os.path.join(data_dir_path, 'cache_regScene_eval')
cache_dir = os.path.join(data_dir_path, 'cache_regScene_eval')
output_dir = os.path.join(data_dir_path, 'attn_eval_output')

temporary_save_dir = os.path.join(data_dir_path, 'temporary_save')

avg_vis_attn_matrix_path = os.path.join(output_dir, 'vis_attn_matrix_average_VISIONS.pt')