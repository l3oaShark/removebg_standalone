import os
from pathlib import Path

import torch
from transformers import AutoModelForImageSegmentation
from PIL import Image
from torchvision import transforms


birefnet = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet-portrait', trust_remote_code=True, cache_dir="./model"
)
torch.set_float32_matmul_precision('highest')
birefnet.to('cpu')
birefnet.eval()


def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cpu')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask


print('start remove background...')

input_dir = Path('.') / 'img' / 'Remove Background 181224'
output_dir = Path('.') / 'output' / 'Remove Background 181224'

all_files = 0
counter = 0

for root, dirnames, filenames in os.walk(input_dir, topdown=True):
    root = Path(root)
    if root.stem == 'output':
        continue
    for dirname in dirnames:
        all_files += len(list((root / dirname).glob('*.jpg')))
        os.makedirs(output_dir / dirname, exist_ok=True)

    for filename in filenames:
        if (root / filename).suffix.lower() == '.jpg':
            input_file = root / filename
            img_dir = output_dir / (root / filename).parts[-2]
            output_file = img_dir / filename

            img = extract_object(birefnet, imagepath=input_file)[0]
            c_img = Image.new('RGB', img.size, (255, 255, 255))
            c_img.paste(img, mask=img)
            c_img.save(output_file)
            # os.remove(input_file)

            counter += 1
            print(f'{counter}/{all_files}')
