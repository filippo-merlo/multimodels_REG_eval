import torch

def load_model(model_name, device, model_dir, cache_dir):
    hf_or_manual = model_name[0]
    model_name = model_name[1]

    if hf_or_manual == 'hf':
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
    else: 
        pass
    
    # BLIP-3
    if model_name == 'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5':
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = model.update_special_tokens(tokenizer)
    
    def get_input(prompt):
        s = (
                    '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                    "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                    f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
                )
        return s
    
    tokenizer.padding_side = "left"
    tokenizer.eos_token = '<|end|>'

    
    return model, get_input