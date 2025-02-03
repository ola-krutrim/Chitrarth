import torch
import base64
import argparse

from chitrarth.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from chitrarth.conversation import conv_templates, SeparatorStyle
from chitrarth.utils import disable_torch_init
from chitrarth.mm_utils import tokenizer_image_token, get_model_name_from_path#, KeywordsStoppingCriteria
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from chitrarth.model.builder import load_pretrained_model
from chitrarth.mm_utils import get_model_name_from_path
import torch
from chitrarth.model import *

from PIL import Image
import os
import shutil

import requests
from PIL import Image
from io import BytesIO

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

        # print("Stopping Criteria keyowrds: ", self.keywords)

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            # print("Stopping Criteria: In if")
            return False
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            flag = False
            
            # print("Stopping Criteria: In else")
            for output in outputs:
                for keyword in self.keywords:
                    # print(f"output: {output}, keyword: {keyword}, all keywords: {self.keywords}")
                    if keyword in output:
                        return True
            return flag

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file))).convert('RGB')
    return image

def eval_model(tokenizer, model, image_processor, context_len, query, image_file, sep=','):

    # disable_torch_init()
    
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "krutrim"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    prompt += ": "
    prompt = prompt.strip()
    print("prompt ", {prompt})
    if image_file:
        image = load_image(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    else:
        image_tensor = None

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = "###"#conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = ["###", "</s>"]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.9,
            top_k=200,
            top_p=0.9,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

# model_base = None
# model_path = "../../llava-responder_v2_mpt-finetune-vit_336_krtrim_conv_hin"

# query = "What is this image about?"
# image_file = "../images/llava_logo.png"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=args.model_base, model_name='mpt_llava')

    eval_model(tokenizer, model, image_processor, context_len, args.query, args.image_file)
