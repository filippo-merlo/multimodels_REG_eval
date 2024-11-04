import os
import json
from config import dataset_path, images_path

def load_dataset():
    # Load and preprocess data
    with open(dataset_path, 'r') as f:
        return json.load(f)

def get_images():
    # Load and preprocess images
    images = {}
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            images[filename] = os.path.join(images_path, filename)
    return images

### CHECK BBOXES
import cv2
save_path = os.path.join('/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data', 'check_bbox')
def save_with_box(image_path, bbox, save_path):
    # load image
    image = cv2.imread(image_path)
    
    # draw box
    print(bbox)
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    
    # save image
    cv2.imwrite(save_path, image)

dataset = load_dataset()
images = get_images()
i = 0
for image_name, image_path in images.items():
    if i > 100:
        break
    save_with_box(image_path, dataset[image_name]['target_bbox'], os.path.join(save_path, image_name))
    i += 1



### ADD NOISE
from PIL import Image
import numpy as np 
def add_gaussian_noise_in_bbox(image_path, bbox, noise_level=0.0):
    # Add noise to the image within the bounding box
    image = Image.open(image_path)
    image_np = np.array(image)
    # box notation [x,w,h,y]
    x, w, h, y = bbox

    return image
