from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from flask import Flask, request, jsonify, send_file
import os
from io import BytesIO

# Model initialization
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True, cache_dir='model')
torch.set_float32_matmul_precision('high')
birefnet.to('cpu')
# birefnet.to('cuda')
birefnet.eval()

app = Flask(__name__)

def extract_object(image, background_color):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0).to('cpu')
    # input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    # Replace background with the provided color
    image = image.convert("RGBA")
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if image.getpixel((x, y))[3] == 0:  # Transparent pixels
                image.putpixel((x, y), background_color)

    return image

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Get RGB values from the request
    try:
        r = int(request.form.get('R', 255))
        g = int(request.form.get('G', 255))
        b = int(request.form.get('B', 255))
        background_color = (r, g, b, 255)  # Add alpha channel
    except ValueError:
        return jsonify({'error': 'Invalid RGB values'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # Process image
    processed_image = extract_object(image, background_color)

    # Save to bytes and send back
    output = BytesIO()
    processed_image.save(output, format="PNG")
    output.seek(0)
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
