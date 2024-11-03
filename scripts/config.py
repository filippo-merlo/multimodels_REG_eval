import os 

data_dir_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data'

# Path of the dictionary with the data 
dataset_path = os.path.join(data_dir_path, 'final_dataset.json')

# Path of the folder with the images
images_path = os.path.join(data_dir_path, 'generated_images')
new_images_path = os.path.join(data_dir_path, 'resized_images')
# Ensure the output directory exists
os.makedirs(os.path.dirname(new_images_path), exist_ok=True)
