import os
import random
#from typing import Optional
from typing import Callable, List, Optional, Union
from torchvision import transforms

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StoryGenPipeline
import argparse

from question_generation.pipelines import pipeline as QG_pipeline

import time

logger = get_logger(__name__)

def get_parser():
    tmp_ref_prompt = "As the sun began to set, a sense of calm settled over the house, signaling that it was time to wind down for the night."
    tmp_prompt = [ 
        "In cozy pajamas, I changed into comfortable sleepwear, ready to snuggle up for a good night's rest.",
        "Heading to the bathroom, I brushed my teeth meticulously, ensuring a clean and fresh feeling before bed.",
        "Back in my room, I dimmed the lights and set a peaceful atmosphere, making it a routine to read a few pages of a book to relax my mind.",
        "After putting down the book, I turned off the lights and said goodnight to my family, ready to drift into a restful sleep.",
        "With a final stretch and a sigh, I settled into bed, surrounded by soft blankets and pillows. The day's events gently faded away as I closed my eyes, ready to embrace a night of restorative sleep."
    ]


    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    #parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-v1-5/', type=str)
    parser.add_argument('--ckpt', default='/home/user/60205048_이혜연/StoryGen/stage2_log/checkpoint_40000', type=str)
    #parser.add_argument('--ref_prompt', default='Mom applies a pea-sized amount of toothpaste onto a soft toothbrush.', type=str)
    #parser.add_argument('--prompt', default=['One day, the white cat is running in the rain.'], type=str)
    parser.add_argument('--ref_prompt', default=tmp_ref_prompt, type=str)
    parser.add_argument('--prompt', default=tmp_prompt, type=List[str])
    parser.add_argument('--num_inference_steps', default=40, type=int)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    ref_prompt: str,
    prompt: Union[str, List[str]],
    num_inference_steps: int = 40,
    guidance_scale: float = 1.0,  #
    mixed_precision: Optional[str] = "no"   # "fp16"
):
    
    # print('ref prompt')
    # print(type(ref_prompt))
    # print('prompt')
    # print(type(prompt))
    # time.sleep(5)

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    # Here, just load the pre-trained SDM weights and initialize the StoryGen model to test the code works well.
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="CLIP")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="CLIP")
    # unet = UNet2DConditionModel.from_config(pretrained_model_path, subfolder="unet")
    # unet.load_SDM_state_dict(torch.load("./ckpt/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin", map_location="cpu"))
    
    # Actually, after training StoryGen, comment out the codes above, and use the following codes to load StoryGen for inference.
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")  
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    
    pipeline = StoryGenPipeline(
        vae=vae,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    visual_projection = image_encoder.visual_projection
    
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")

    vae.eval()
    text_encoder.eval()
    image_encoder.eval()
    unet.eval()
    
    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    ref_output = pipeline(
        cond = None,
        prompt = ref_prompt,
        height = 512,
        width = 512,
        generator = generator,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        num_images_per_prompt = 1,
        cross_frame_attn = False,
    )

    # visualize noise and image
    ref_image = ref_output.images[0][0] # PIL Image here
    ref_image.save(os.path.join(logdir, "image0.png"))
    ref_image = ref_image.resize((224, 224))
    ref_image = transforms.ToTensor()(ref_image)
    ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
    ref_image = ref_image * 2. - 1.
    ref_image = ref_image.unsqueeze(0)
    ref_image = ref_image.to(accelerator.device)
    
    ref_img_feature = image_encoder(ref_image).last_hidden_state
    projected_ref_img_feature = visual_projection(ref_img_feature)

    i = 0
    for prompt_ in prompt:

        print(prompt_)
        output = pipeline(
            cond = projected_ref_img_feature,
            prompt = prompt_,
            height = 512,
            width = 512,
            generator = generator,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            num_images_per_prompt = 1,
            cross_frame_attn = True,
        )

        ref_image = output.images[0][0] # PIL Image here
        ref_image.save(os.path.join(logdir, "image{}.png".format(i)))
        ref_image = ref_image.resize((224, 224))
        ref_image = transforms.ToTensor()(ref_image)
        ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
        ref_image = ref_image * 2. - 1.
        ref_image = ref_image.unsqueeze(0)
        ref_image = ref_image.to(accelerator.device)
    
        ref_img_feature = image_encoder(ref_image).last_hidden_state
        projected_ref_img_feature = visual_projection(ref_img_feature)
        i += 1

    #output_image = output.images[0][0] # PIL Image here
    #output_image.save(os.path.join(logdir, ".png"))

def QG_test(
    ref_prompt : str,
    prompt: Union[str, List[str]]
):
    i = 0
    tmp = open(os.path.join(logdir, 'image{}.txt').format(i), 'w')
    tmp.write(ref_prompt)
    tmp.close()
    story = ref_prompt
    for line in prompt:
        i += 1
        tmp = open(os.path.join(logdir, 'image{}.txt').format(i), 'w')
        tmp.write(line)
        tmp.close()
        story += ' ' + line
    nlp = QG_pipeline("e2e-qg", model="valhalla/t5-base-e2e-qg")
    print('story : ', story)
    question_list = nlp(story)
    print('question : ', question_list)
    txt = open(os.path.join(logdir, 'questions.txt'), 'w')
    for question in question_list:
        txt.write(question + '\n')
    txt.close()        


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    ref_prompt = args.ref_prompt
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    mixed_precision = "no" # "fp16",
    test(pretrained_model_path, logdir, ref_prompt, prompt, num_inference_steps, guidance_scale, mixed_precision)
    QG_test(ref_prompt, prompt)


# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py