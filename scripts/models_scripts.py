import torch
# prompt base: What is the object in this part of the image <bbox>?

def load_model(model_name, device, model_dir, cache_dir):
    '''
    hf_or_manual = model_name[0]
    model_name = model_name[1]

    if hf_or_manual == 'hf':
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
    else: 
        pass
    '''

    from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

    # BLIP-3
    if model_name == 'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5':
        
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

            x, y, w, h = map(int, bbox)

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            prompt = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\nWhat is the object in this part of the image <bbox>{x1}, {y1}, {x2}, {y2}</bbox><|end|>\n<|assistant|>\n'
            )

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
        
        # Load the model, tokenizer, and image processor
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        
        # Move model to the device
        model = model.to(device)
        model.eval()
        
        # Update special tokens for the tokenizer
        tokenizer = model.update_special_tokens(tokenizer)
        
        def apply_prompt_template(prompt):
            s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
            )
            return s
        
        class EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence=[32007]):
                self.eos_sequence = eos_sequence

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids  

        def generate(model, image, prompt):
            # Preprocess the image
            inputs = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')

            # Apply the prompt template
            prompt_with_template = apply_prompt_template(prompt)

            # Tokenize the prompt and prepare inputs
            language_inputs = tokenizer([prompt_with_template], return_tensors="pt")
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

    elif model_name == '':
        # other models
        return model, generate
    else:
        # other models
        return model, generate
