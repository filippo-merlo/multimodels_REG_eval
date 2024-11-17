import torch
from PIL import Image
from utils import convert_box, normalize_box, convert_bbox_to_point
# prompt base: What is the object in this part of the image <bbox>?


def load_model(model_name, device, model_dir, cache_dir):

    # BLIP-3
    if model_name == 'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5':
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        
        model = model.to(device)
        model.eval()
        
        tokenizer = model.update_special_tokens(tokenizer)
        tokenizer.padding_side = "left"
        tokenizer.eos_token = '<|end|>'
        
        def generate(model, image, bbox):
            pixel_values = image_processor([image], image_aspect_ratio='anyres')["pixel_values"].cuda()

            inputs = {
                "pixel_values": [pixel_values]
            }

            W = image.size[0]
            H = image.size[1]
            normalized_bbox = normalize_box(convert_box(bbox), W, H)
            x1, y1 ,x2, y2 = normalized_bbox

            prompt = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>What is the object in this part <bbox>[{x1}, {y1}][{x2}, {y2}]</bbox> of the image? Can be Nothing.<|end|>\n<|assistant|>\n'
            ) # add image in the promt 

            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)

            # Move tensors to CUDA
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[name] = value.cuda()

            generated_text = model.generate(**inputs, image_size=[image.size],  
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            temperature=0.05,
                                            do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
            
            prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
            return prediction

        return model, generate 
    
    if model_name == 'Salesforce/xgen-mm-phi3-mini-instruct-r-v1':
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
        # Load the model, tokenizer, and image processor
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        
        # Move model to the device
        model = model.to(device)
        model.eval()
        
        # Update special tokens for the tokenizer
        tokenizer = model.update_special_tokens(tokenizer)
        
        class EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence=[32007]):
                self.eos_sequence = eos_sequence

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids  

        def generate(model, image, bbox):
            # Preprocess the image
            inputs = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')

            pixel_values = image_processor([image], image_aspect_ratio='anyres')["pixel_values"].cuda()
            pixel_values = inputs["pixel_values"].squeeze(0).cuda()  # Remove unnecessary batch dimension
    
            inputs = {
                "pixel_values": pixel_values.unsqueeze(0)  # Re-add batch dimension if needed for model input
            }

            W = image.size[0]
            H = image.size[1]
            normalized_bbox = normalize_box(convert_box(bbox), W, H)
            x1, y1 ,x2, y2 = normalized_bbox

            prompt = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>What is the object in this part <bbox>[{x1}, {y1}][{x2}, {y2}]</bbox> of the image? Can be Nothing.<|end|>\n<|assistant|>\n'
            ) # add image in the promt 

            # Tokenize the prompt and prepare inputs
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)

            # Move tensors to the device (GPU or CPU)
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[name] = value.to(device)

            # Generate text with the model
            generated_text = model.generate(**inputs, image_size=[image.size],
                                            pad_token_id=tokenizer.pad_token_id,
                                            do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                            stopping_criteria=[EosListStoppingCriteria()])

            # Decode and return the generated text
            prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
            return prediction
        
        return model, generate 

    elif model_name == 'microsoft/kosmos-2-patch14-224':
        # Load Kosmos-2 model and processor
        from transformers import Kosmos2ForConditionalGeneration, AutoProcessor
        
        model = Kosmos2ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_dir)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        model = model.to(device)
        model.eval()

        def generate(model, image, bbox):
            # Adapt box
            W = image.size[0]
            H = image.size[1]
            normalized_bbox = normalize_box(convert_box(bbox), W, H)

            prompt="<grounding>What is the object in <phrase>this part</phrase> of the image?"

            # Preprocess the image and prompt
            inputs = processor(images = [image], text = [prompt],  bboxes = [[normalized_bbox]] , return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=64,
            )

            # Decode generated output and process text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
            
            return processed_text
        
        return model, generate


    elif model_name == 'allenai/Molmo-7B-O-0924':
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        # Load the processor and model
        processor = AutoProcessor.from_pretrained( model_name, trust_remote_code=True, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained( model_name, trust_remote_code=True, cache_dir=model_dir)
        model = model.to(device)
        model.eval()

        def generate(model, image, bbox):
            x1, y1 = convert_bbox_to_point(bbox)
            prompt=f"What is the object at point x = {x1}, y = {y1} of the image?"

            # Process image from URL and text prompt
            inputs = processor.process(
                images=[image],
                text=prompt
            )

            # Move inputs to the correct device and create a batch of size 1
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            # Generate output with configured generation settings
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
                tokenizer=processor.tokenizer
            )

            # Decode generated tokens to text
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text

        return model, generate
    
    if model_name == 'jadechoghari/Ferret-UI-Gemma2b':
        from ferret_ui.inference import inference_and_run
        
        def generate(model, image, bbox):
            model = ''
            prompt = 'What is the object in this part of the image?'
            generated_text = inference_and_run(
                image=image, 
                prompt=prompt, 
                conv_mode="ferret_gemma_instruct", 
                model_path=model_name, 
                box=map(int(convert_box(bbox)))
            )
            return generated_text

        model = None
        return model, generate
    else:
        # other models
        return model, generate
