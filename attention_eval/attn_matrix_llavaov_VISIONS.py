#%%
# Import  
'''
For this experiment I have used a sligtly modified version of LLaVA-NeXT (to allow for the extraction of attention maps from the visual backbone), available at:
git clone https://github.com/filippo-merlo/LLaVA-NeXT.git
cd LLaVA-NeXT
conda create -n myenv python=3.10 -y
conda activate myenv
pip install --upgrade pip
pip install -e ".[train]"
'''
import os
import gc
import copy
from tqdm import tqdm
from config_attn_analysis import *
from utils_attn_analysis import *

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


# load the model
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

load_8bit = False
load_4bit = False

llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
    "load_8bit" : load_8bit,
    "load_4bit" : load_4bit,
}

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args, cache_dir=cache_dir, device_map=device_map)
model.eval()

# Load the CSV with the bounding box data
bbox_data = get_bbox_data(dataset_path)

# Select which images to keep
images_n_p_full = get_images_names_path(images_path)
include = ['l', 'r']#, 'sl', 'sr']
images_n_p = {}
for image_name, image_path in get_images_names_path(images_path).items():
    if image_name.split('_')[3] in include:
        images_n_p[image_name.replace('.jpg', '')] = image_path

noise_levels = [0.0]
conditions = ['target_noise']

vis_attn_matrix_average = []

for condition in conditions:
  for noise_level in noise_levels:
    for image_name, image_path  in tqdm(list(images_n_p.items())):

      bbox = bbox_data[bbox_data['scene'] == image_name]['bbox'].values[0]
      target = image_name.split('_')[4]

      # get the image with a grey background and the bounding box rescaled
      # in this function the image is also resized to have a maximum w of 640px
      image, bbox, original_image_size = rescale_image_add_grey_background_and_rescale_bbox(image_path, bbox, 640)
      print(f"Bbox: {bbox}")
      image_patch = get_image_patch(image, bbox)

      temporary_save_path_image_patch = os.path.join(temporary_save_dir,f'image_patch_{image_name}.jpg')
      if not os.path.exists(temporary_save_path_image_patch):
          image_patch.save(temporary_save_path_image_patch)

      # get the image with the corresponding noise level in the roi
      if condition == 'target_noise':
          image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
      elif condition == 'context_noise':
          image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)
      elif condition == 'all_noise':
          image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
          image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)

      # Process input
      image_sizes = image.size
      W = image.size[0]
      H = image.size[1]
      normalized_bbox = normalize_box(convert_box(bbox), W, H)
      print(f"Normalized bbox: {normalized_bbox}")
      x1, y1, x2, y2 = normalized_bbox


      with torch.inference_mode():

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor] ##

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        prompt_text = DEFAULT_IMAGE_TOKEN + f"\nWhat is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # generate the response
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_sizes],
            do_sample=False,
            temperature=0,
            max_new_tokens=10,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

        # connect with the vision encoder attention
        # to visualize the attention over the image

        # vis_attn_matrix will be of torch.Size([N, N])
        # where N is the number of vision tokens/patches
        # `all_prev_layers=True` will average attention from all layers until the selected layer
        # otherwise only the selected layer's attention will be used

        att_on_whole_image = []

        for v in model.get_vision_tower().image_attentions:
          att_on_whole_image.append(v[0])
          del v

        del image_tensor, input_ids, outputs, model.get_vision_tower().image_attentions

        vis_attn_matrix_per_layers = []

        for layer in list(range(0,26)):

          vis_attn_matrix = aggregate_vit_attention(
              att_on_whole_image,
              select_layer=layer
          )
          vis_attn_matrix_per_layers.append(vis_attn_matrix)
          del vis_attn_matrix
        
        del att_on_whole_image

        if vis_attn_matrix_average == []:
            vis_attn_matrix_average = torch.stack(vis_attn_matrix_per_layers)
            for v in vis_attn_matrix_per_layers:
                del v
            del vis_attn_matrix_per_layers
        else:
            two_tensors = torch.stack([vis_attn_matrix_average, torch.stack(vis_attn_matrix_per_layers)])
            vis_attn_matrix_average = torch.mean(two_tensors, dim=0)
            for v in vis_attn_matrix_per_layers:
                del v
            del vis_attn_matrix_per_layers, two_tensors

        gc.collect()
        torch.cuda.empty_cache()

print(vis_attn_matrix_average)
print(vis_attn_matrix_average.size())
output_tensor_path = avg_vis_attn_matrix_path
torch.save(vis_attn_matrix_average, output_tensor_path)
print(f"vis_attn_matrix_average saved at {output_tensor_path}")
