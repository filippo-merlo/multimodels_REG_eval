import os
import json
from config import data_dir_path, dataset_path, images_path

def load_dataset():
    # Load and preprocess data
    with open(dataset_path, 'r') as f:
        return json.load(f)

def get_images_names_path():
    # Load and preprocess images
    images_n_p = {}
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            images_n_p[filename] = os.path.join(images_path, filename)
    return images_n_p


dataset = load_dataset()
i = 0 
for k, v in dataset.items():
    if i > 10:
        break
    print(k, v)
    i += 1
'''
images = get_images_names_path()

i = 0
os.makedirs(os.path.join(data_dir_path,'noisy_images'))
for image_name, image_path in images.items():
    i += 1
    if i > 100:
        break
    bbox = dataset[image_name]['target_bbox']
    noisy_image = add_gaussian_noise_in_bbox(image_path, bbox, noise_level=0.5)
    noisy_image.save(os.path.join(data_dir_path, f'noisy_images/{image_name}'))
    '''