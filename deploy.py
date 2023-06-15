import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )

    return model

def model_fn(model_dir):
       
    model = net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('Model loaded successfully.')

    model.eval()

    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception(f'Unsupported ContentType: {content_type}')

def predict_fn(input_object, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction