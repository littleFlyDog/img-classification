from json import load
from loguru import logger
import torch
from clsCLIP.model import ClsAny
import argparse
from data import load_image

def main(args):
    logger.info('=============== Starting ================')
    kwargs=vars(args)
    device = torch.device(f'cuda:{kwargs.get("gpu_id", 0)}' if torch.cuda.is_available() else 'cpu')
    kwargs['device']=device
    logger.info(f'load_image')
    #load image
    img=load_image(**kwargs)
    #create model ,创建好了model
    logger.info(f'create model')
    model = ClsAny(**kwargs)
    model = model.to(device)
    test(img,model,device)
    #test

def test(img,model,device):
    model.eval()
    with torch.no_grad():
        model(img)
        while True:
            print("please input your jugement:,you can exit by inputing 0")
            input_text = input()
            if input_text == '0':
                logger.info('=============== Exiting ================')
                break
            model.build_text_feature_gallery(input_text)
            #calculate anomaly score
            model.calculate_textual_anomaly_score()


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--imgdir', type=str,default='./test1.jpg',help='image directory')
    parser.add_argument('--gpu-id', type=int,default=0,help='gpu device')
    parser.add_argument('--img-resize', type=int,default=240,help='image resize')
    parser.add_argument('--img-cropsize', type=int, default=240, help='image crop size')
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    
    args = parser.parse_args()

    return args




if __name__ == '__main__':
    args = get_args()
    main(args)
