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
import openai
import shutil

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

logger = get_logger(__name__)

def get_parser():
    # tmp_ref_prompt = "As the sun began to set, a sense of calm settled over the house, signaling that it was time to wind down for the night."
    # tmp_prompt = [ 
    #     "In cozy pajamas, I changed into comfortable sleepwear, ready to snuggle up for a good night's rest.",
    #     "Heading to the bathroom, I brushed my teeth meticulously, ensuring a clean and fresh feeling before bed.",
    #     "Back in my room, I dimmed the lights and set a peaceful atmosphere, making it a routine to read a few pages of a book to relax my mind.",
    #     "After putting down the book, I turned off the lights and said goodnight to my family, ready to drift into a restful sleep.",
    #     "With a final stretch and a sigh, I settled into bed, surrounded by soft blankets and pillows. The day's events gently faded away as I closed my eyes, ready to embrace a night of restorative sleep."
    # ]

    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    # parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-v1-5/', type=str)
    parser.add_argument('--ckpt', default='/home/user/60205048_이혜연/StoryGen/stage2_log/checkpoint_40000', type=str)
    # parser.add_argument('--ref_prompt', default='Mom applies a pea-sized amount of toothpaste onto a soft toothbrush.', type=str)
    # parser.add_argument('--prompt', default=['One day, the white cat is running in the rain.'], type=str)
    # parser.add_argument('--ref_prompt', default=tmp_ref_prompt, type=str)
    # parser.add_argument('--prompt', default=tmp_prompt, type=List[str])
    parser.add_argument('--num_inference_steps', default=40, type=int)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    parser.add_argument('--story_prompt', default=None, type=str)
    parser.add_argument('--num_scenes', default=6, type=int)

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

    folder_list = ['image', 'prompt', 'question', 'subtitle', 'title']
    for folder in folder_list:
        if os.path.exists(os.path.join(logdir, folder)):
            shutil.rmtree(os.path.join(logdir, folder))
        os.mkdir(os.path.join(logdir, folder))   
           
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

    # save_prompt = open(os.path.join(logdir, 'image0.txt'), 'w')
    # save_prompt.write(ref_prompt)
    # save_prompt.close()

    # visualize noise and image
    ref_image = ref_output.images[0][0] # PIL Image here
    ref_image.save(os.path.join(logdir, 'image/image0.png'))
    ref_image = ref_image.resize((224, 224))
    ref_image = transforms.ToTensor()(ref_image)
    ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
    ref_image = ref_image * 2. - 1.
    ref_image = ref_image.unsqueeze(0)
    ref_image = ref_image.to(accelerator.device)
    
    ref_img_feature = image_encoder(ref_image).last_hidden_state
    projected_ref_img_feature = visual_projection(ref_img_feature)

    if isinstance(prompt, str):
        prompt = list(prompt)
        
    
    for prompt_, i in zip(prompt, range(1, len(prompt) + 1)):
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

        # save_prompt = open(os.path.join(logdir, 'image{}.txt'.format(i)), 'w')
        # save_prompt.write(prompt_)
        # save_prompt.close()

        ref_image = output.images[0][0] # PIL Image here
        ref_image.save(os.path.join(logdir, 'image/image{}.png'.format(i)))
        ref_image = ref_image.resize((224, 224))
        ref_image = transforms.ToTensor()(ref_image)
        ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
        ref_image = ref_image * 2. - 1.
        ref_image = ref_image.unsqueeze(0)
        ref_image = ref_image.to(accelerator.device)

        ref_img_feature = image_encoder(ref_image).last_hidden_state
        projected_ref_img_feature = visual_projection(ref_img_feature)
    # end for


def QG_test(ref_prompt : str, prompt: Union[str, List[str]], logdir: str):    
    if isinstance(prompt, str):
        prompt = list(prompt)

    story = ref_prompt
    for line in prompt:
        story += ' ' + line

    nlp = QG_pipeline("e2e-qg", model="valhalla/t5-base-e2e-qg")
    question_list = nlp(story)

    # for question, i in zip(question_list, range(len(question_list))):
    #     txt = open(os.path.join(logdir, 'question{}.txt'), 'w')
    #     txt.write(question)
    #     txt.close()
    return question_list


def make_content(story_prompt : str, num_scenes : int):
    content = story_prompt 
    content += ' 그리고 그 이야기를 {}장면으로 나눠줘 영어로만 보여줘.'.format(num_scenes) 
    content += ' 그리고 각 장면의 문장은 최대 77토큰으로 만들어줘.'
    content += ''' 
    각 장면은 
    Scene: 장면 번호
    Subtitle: 소제목
    Content: 장면 내용
    으로 출력해줘
    '''
    content += ' 한글 이야기는 출력하지 마.'
    content += ' 첫 줄에는 이야기 제목을 출력해줘.'
    return content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_api(prompt : str, option : int, num_scenes = None):
    openai_key = "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" 
    openai.api_key = openai_key
    openai.api_base = "https://api.openai-proxy.com/v1"
    messages = list()

    # make_story
    if option == 0:
        content = make_content(prompt, num_scenes)
    # translation
    elif option == 1:
        content = prompt + ' 앞 문장을 한국어로 번역해줘. 번역한 내용만 출력해줘.'
    elif option == 2:
        content = prompt + ' 의역해 줘. 명사는 최대한 한글로 출력해줘.'
    elif option == 3:
        content = prompt + ' 앞 문장을 한국어로 번역해줘. 번역한 내용만 출력해줘.'    

    messages.append({'role':'user', 'content':content})
    response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo', messages = messages)
    reply = response['choices'][0]['message']['content']   
    print('gpt_reply: ', reply)
    # total_tokens = response['usage']['total_tokens']
    # return reply, total_tokens
    return reply


def story_parser(story : str):
    storyList = story.split('\n')
    title = str()
    prompts = list()
    subtitles = list()

    for i in range(len(storyList)):
        line = storyList[i]
        if line.find('Title') >= 0:
            title = line.replace('Title: ', '')
        elif line.find('Scene') >= 0:
            continue
        elif line.find('Subtitle') >= 0:
            subtitles.append(line.split(': ')[1])
        elif line.find('Content') >= 0:
            prompts.append(line.split(': ')[1])

    ret_dict = {'title':title, 'prompts':prompts, 'subtitles':subtitles}
    return ret_dict


def save_items(title, prompts, subtitles, questions, logdir):
    title_file = open(os.path.join(logdir, 'title/title.txt'), 'w')
    title = gpt_api(title, 2)
    title_file.write(title)
    title_file.close()

    def save_item(item_list, item_name, option):
        for item, i in zip(item_list, range(len(item_list))):
            item_path = os.path.join(logdir, item_name + '/' + item_name + str(i) + '.txt')
            item_file = open(os.path.join(item_path), 'w')
            item = gpt_api(item, option)
            item_file.write(item)
            item_file.close()
    
    save_item(prompts, 'prompt', 1)
    save_item(subtitles, 'subtitle', 2)
    save_item(questions, 'question', 3)

    # for subtitle, i in zip(subtitles, range(len(subtitles))):
    #     subtitle_file = open(os.path.join(logdir, 'subtitle/subtitle{}.txt'.format(i)))
    #     subtitle_file.write(subtitle)
    #     subtitle_file.close()
    # for question, i in zip(questions, range(len(questions))):
    #     question_file = open(os.path.join(logdir, 'question/question{}.txt'.format(i)))
    #     question_file.write(question)
    #     question_file.close() 

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    mixed_precision = "no" # "fp16",
    story_prompt = args.story_prompt
    num_scenes = args.num_scenes

    story = gpt_api(story_prompt, 0, num_scenes)
    # print('story', story)
    
    parser_dict = story_parser(story)
    prompts = parser_dict['prompts']

    # print('prompts : ', prompts)
    ref_prompt = prompts[0]
    prompt = prompts[1:]

    test(pretrained_model_path, logdir, ref_prompt, prompt, num_inference_steps, guidance_scale, mixed_precision)
    questions = QG_test(ref_prompt, prompt, logdir)

    save_items(parser_dict['title'], parser_dict['prompts'], parser_dict['subtitles'], questions, logdir)

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py