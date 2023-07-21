import torch
from fastapi import FastAPI, Request, Form, File, UploadFile
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
import torch.nn as nn
import io

app = FastAPI()

model = models.resnet50(weights='IMAGENET1K_V1').to('cpu')
def img_transfromer(img):
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    }
    img_input = data_transforms['test'](img)
    #transformation에서 추가적인 차원추가로 4D 텐서로 변환
    img_input.unsqueeze_(0)
    return img_input



# 파라미터를 수정하지 않도록 설정.
for param in model.parameters():
    param.requires_grad = False


# 모델의 수정
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 18)
)

model.load_state_dict(torch.load('C:/eadgnus/talent_picture.pth', map_location=torch.device('cpu')))
model.eval()


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    }
# dataset 생성
image_datasets = {
    'test': datasets.ImageFolder('./train/', data_transforms['test'])
}

@app.post("/")
async def show_img(file: UploadFile = File(...)):
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents))
    plt.imshow(np.array(img_pil))
    plt.axis('off')
    plt.show()

@app.post("/test")
async def predic(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    img_input = img_transfromer(img)

    y_pred = model(img_input)
    y_prob = nn.Softmax(1)(y_pred)

    probs, indices = torch.topk(y_prob, k=3, dim=-1)  # (2,3,224,224)에서 dim=-1은 224

    probs = probs.cpu().data.numpy()
    indices = indices.cpu().data.numpy()

    fig, axes = plt.subplots(1, 1, figsize=(20, 10))

    axes.set_title('{:.2f}% {}\n{:.2f}% {}\n{:.2f}% {}'.format(
        probs[0, 0] * 100, image_datasets['test'].classes[indices[0, 0]],
        probs[0, 1] * 100, image_datasets['test'].classes[indices[0, 1]],
        probs[0, 2] * 100, image_datasets['test'].classes[indices[0, 2]]
    ))
    axes.imshow(np.array(img))
    axes.axis('off')
    plt.show()