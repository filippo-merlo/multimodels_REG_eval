from PIL import Image, ImageDraw
from utils import convert_box, normalize_box, convert_bbox_to_point, normalize_box_N, normalize_box_cogvlm
import os
# prompt base: What is the object in this part of the image <bbox>?


def load_model(model_name, device, model_dir, cache_dir):
    import torch
    # BLIP-3
    if model_name == 'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5':
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        model.to(device)
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
                "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f"<|user|>\n<image>What is the object in this part <bbox>[{x1}, {y1}][{x2}, {y2}]</bbox> of the image? Answer with the object's name only. No extra text.<|end|>\n<|assistant|>\n"
            ) # add image in the promt 

            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            #decoded_input = tokenizer.decode(inputs['input_ids'][0])


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
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True,cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        
        # Move model to the device
        model.to(device)
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
                "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f"<|user|>\n<image>What is the object in this part <bbox>[{x1}, {y1}][{x2}, {y2}]</bbox> of the image? Answer with the object's name only. No extra text.<|end|>\n<|assistant|>\n"
            ) # add image in the promt 

            # Tokenize the prompt and prepare inputs
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            decoded_input = tokenizer.decode(inputs['input_ids'][0])

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
        
        model = Kosmos2ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map='auto', cache_dir=model_dir)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        model.eval()

        def generate(model, image, bbox):
            # Adapt box
            W = image.size[0]
            H = image.size[1]
            normalized_bbox = tuple(normalize_box(convert_box(bbox), W, H))

            prompt="<grounding>What is the object in <phrase>this part</phrase> of the image? Answer with the object's name only. No extra text. Answer:"

            # Preprocess the image and prompt
            inputs = processor(images = [image], text = [prompt],  bboxes = [[normalized_bbox]] , return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            #decoded_input = processor.decode(inputs['input_ids'][0])

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
            processed_text, entities = processor.post_process_generation(generated_text)
            
            return processed_text.replace("What is the object in this part of the image? Answer with the object's name only. No extra text. Answer: ", "")
        
        return model, generate

    elif model_name == 'cyan2k/molmo-7B-O-bnb-4bit':
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            GenerationConfig,
        )
        from PIL import Image

        # Load the processor and model
        arguments = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            **arguments, 
            cache_dir=cache_dir)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **arguments,
            cache_dir=model_dir)
        
        model.eval()

        def generate(model, image, bbox):
            W, H = image.size
            x1, y1 = convert_bbox_to_point(normalize_box_N(bbox, W, H, 100))
            print('point:',x1, y1)
            
            prompt=f"What is the object at point x = {int(x1)}, y = {int(y1)} of the image? Answer with the object's name only. No extra text."
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                    # Process image from URL and text prompt
                    inputs = processor.process(
                        images=[image],
                        text=prompt
                    )

                    # Move inputs to the correct device and create a batch of size 1
                    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
                    #decoded_input = processor.tokenizer.decode(inputs['input_ids'][0])
            
                    # Generate output with configured generation settings
                    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
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
    
    elif model_name == "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8":
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info

        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            cache_dir=model_dir,
        )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # default processer
        processor = AutoProcessor.from_pretrained(model_name,cache_dir=cache_dir)

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", min_pixels=min_pixels, max_pixels=max_pixels)

        def generate(model, image, bbox):
            x1, y1, x2, y2 = map(lambda x: round(x, 1), convert_box(normalize_box_N(bbox))) 

            path = "temp_image.jpg"
            image.save(path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": path,
                        },
                        {
                            "type": "text",
                            "text": f"What is the object in <|object_ref_start|>this part of the image<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>? Answer with the object's name only. No extra text."
                        },
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            #decoded_input = processor.decode(inputs['input_ids'][0])

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            os.remove(path)
            return output_text[0].replace(f"What is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text.assistant",'')
 
        return model, generate
    
    elif model_name == "THUDM/cogvlm2-llama3-chat-19B-int4":
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=model_dir,
            device_map='auto',
        ).eval() # Load the model and set it to evaluation mode

        def generate(model, image, bbox):

            x1, y1, x2, y2 = normalize_box_cogvlm(convert_box(bbox))
            
            question = f"What is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text."
            prompt  = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"

            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=prompt,
                images=[image],
                history = [],
                template_version='chat'
            )

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(model.device),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(model.device),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(model.device),
                'images': [[input_by_model['images'][0].to(model.device).to(TORCH_TYPE)]] if image is not None else None,
            }

            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("<|end_of_text|>")[0]
            
            return response    
            
        
        return model, generate

    elif model_name == "llava-hf/llava-onevision-qwen2-0.5b-si-hf":
        from PIL import Image
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            cache_dir=model_dir,
            device_map='auto',
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

        def generate(model, image, bbox):
            # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
            # Each value in "content" has to be a list of dicts with types ("text", "image") 
            W = image.size[0]
            H = image.size[1]
            normalized_bbox = normalize_box(convert_box(bbox), W, H)
            x1, y1, x2, y2 = normalized_bbox
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": f"What is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text."},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            raw_image = image
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            #decoded_input = processor.decode(inputs['input_ids'][0])
          

            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            output_text = processor.decode(output[0][2:], skip_special_tokens=True)

            return output_text.replace(f"What is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text.assistant\n",'').replace('\n','')
        return model, generate
    else:
        # other models
        return model, generate


