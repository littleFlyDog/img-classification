from sympy import N
import torch
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image

valid_backbones = ['ViT-B-16-plus-240']
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

import os

def clear_screen():
    # 如果是 Windows 系统 ('nt')
    if os.name == 'nt':
        os.system('cls')
    # 如果是 Linux 或 Mac 系统 ('posix')
    else:
        os.system('clear')

def _convert_to_rgb(image):
    return image.convert('RGB')

class ClsAny(torch.nn.Module):
    def __init__(self, device, backbone, pretrained_dataset, precision='fp32', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(ClsAny, self).__init__()
        self.precision =  'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop 

        self.device = device
        self.get_model(backbone, pretrained_dataset)
        self.phrase_form = '{}'

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        # self.version = 'V1' # V1:
        # # visual textual, textual_visual
        # self.fusion_version = 'textual_visual'
        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

    def get_model(self, backbone, pretrained_dataset):
        #两个判断
        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets
#此处加载好model
        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval().to(self.device)

        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None
        # print("self.grid_size",self.grid_size)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        # return [f / f.norm(dim=-1, keepdim=True) for f in image_features]
        return image_features / image_features.norm(dim=-1, keepdim=True) 
    

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        return text_features
    
    def build_text_feature_gallery(self, category: str):
        phrases = []



        # some categories can be renamed to generate better embedding
        #if category == 'grid':
        #    category  = 'chain-link fence'
        #if category == 'toothbrush':
        #    category = 'brush' #'brush' #
        for template_prompt in template_level_prompts:
            # normal prompts
            # for normal_prompt in state_level_normal_prompts:
            #     phrase = template_prompt.format(normal_prompt.format(category))
            #     normal_phrases += [phrase]

            # # abnormal prompts
            # for abnormal_prompt in state_level_abnormal_prompts:
            #     phrase = template_prompt.format(abnormal_prompt.format(category))
            #     abnormal_phrases += [phrase]
            phrase= template_prompt.format(category)
            phrases += [phrase]

        phrases = self.tokenizer(phrases).to(self.device)
#至此正常和异常句子的tokenizer完成，每句话已经被转成了长度为77的一维tensor
        

        text_features = self.encode_text(phrases)

        text_features = torch.mean(text_features, dim=0, keepdim=True)


        self.text_features = text_features

        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def calculate_textual_anomaly_score(self):
        score=(self.visual_features@ self.text_features.T)
        # print(self.visual_features.norm())
        # print(self.text_features.norm())
        score=score.item()
        clear_screen()
        print("===================================")
        if score>0.3:
            print("Absolutely correct!")
        elif score>0.22:
            print("Most likely correct.")
        elif score>0.15:
            print("Probably uncorrect.")
        else:
            print("Absolutely uncorrect!")
        print("similarity score:",score)
        print("===================================",'\n')



    def forward(self, images):

        visual_features = self.encode_image(images)#返回的是窗口个[bs,896]的list为clstoken输出
        self.visual_features = visual_features


    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
