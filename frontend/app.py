from flask import Flask, render_template, jsonify, send_file
import numpy as np
import pandas as pd
import random
import cv2
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from supervised import CNNModel

app = Flask(__name__)

# Load trained models
# GAN
generator_model = load_model(os.path.join(project_root, "results/unsupervised/model/generator_model.h5"))

# Supervised Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supervised_model_path = os.path.join(project_root, "results/supervised/best_model.pth")
supervised_model = CNNModel(filters=(32, 64, 128))
supervised_model.load_state_dict(torch.load(supervised_model_path, map_location=device))
supervised_model.to(device)
supervised_model.eval()
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# State-of-the-art model
model_path = 'results/sota/model/final_transfer_model.h5'
sota_model = load_model(os.path.join(project_root, model_path))

# Load reduced image dataset
reduced_image_dir = os.path.join(project_root, 'data/reduced_train/')
labels_path = os.path.join(project_root, 'data/train_labels.csv')
labels_df = pd.read_csv(labels_path)
image_labels = dict(zip(labels_df['id'], labels_df['label']))
image_paths = [os.path.join(reduced_image_dir, f) for f in os.listdir(reduced_image_dir) if f.endswith('.tif')]

# Helper functions
def generate_image():
    noise = np.random.normal(0,1,(1,128))
    generated_image = generator_model.predict(noise, verbose=0)[0]
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
    if generated_image.shape[-1] == 1:
        generated_image = generated_image.squeeze()
    return generated_image

def get_random_image():
    random_image_path = random.choice(image_paths)
    random_image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
    return random_image

def image_to_base(img):
    img_pil = Image.fromarray(img)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_image

def preprocess_sota_image(img):
    img = np.array(img.convert('L'))
    img = cv2.resize(img, (128, 128))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = (img - np.mean(img)) / np.std(img)
    img = img.reshape(1, 128, 128, 1)
    img = np.repeat(img, 3, axis=-1)
    return img.astype('float32')

# API Endpoints
@app.route("/")
def homepage():
    return render_template('home.html')

@app.route("/supervised-model")
def supervisedModel():
    return render_template('supervised.html')

@app.route("/unsupervised-model")
def unsupervisedModel():
    return render_template('unsupervised.html')

@app.route("/sota-model")
def sotaModel():
    return render_template('sota.html')

@app.route('/generate', methods=['GET'])
def generate():
    fake_image = generate_image()
    real_image = get_random_image()
    
    return jsonify({
        'fake': image_to_base(fake_image),
        'real': image_to_base(real_image)
    })

@app.route('/supervised-predictions', methods=['GET'])
def supervisedPrediction():
    image_path = random.choice(image_paths)
    image = Image.open(image_path).convert("RGB")  
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get actual class
    image_id = os.path.basename(image_path).split('.')[0]  
    actual_class = image_labels.get(image_id)

    # Get predicted class
    with torch.no_grad():
        output = supervised_model(image_tensor)
        predicted_class = torch.sigmoid(output).item()
        predicted_class = 1 if predicted_class > 0.5 else 0  

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "image": encoded_img,
        "actual_label": actual_class,
        "predicted_label": predicted_class
    })

@app.route('/state-of-the-art-predictions', methods=['GET'])
def sotaPrediction():
    image_path = random.choice(image_paths) 
    image = Image.open(image_path).convert("RGB")  

    # Get actual class
    image_id = os.path.basename(image_path).split('.')[0]  
    actual_class = image_labels.get(image_id)

    processed_image = preprocess_sota_image(image)
    prediction = sota_model.predict(processed_image)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0 

    # Convert image to base64 
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "image": encoded_img,
        "actual_label": actual_class,
        "predicted_label": predicted_class
    })

# Run Flask App
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8501, debug=True)