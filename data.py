from PIL import Image
import torchvision.transforms as transforms
MEAN=[0.48145466, 0.4578275, 0.40821073]
STD=[0.26862954, 0.26130258, 0.27577711]

def load_image(**kwargs):
    image = Image.open(kwargs.get('imgdir'))
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((kwargs.get('img_resize', 224), kwargs.get('img_resize', 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=MEAN, std=STD)
    # ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image=image.to(kwargs.get('device'))
    return image