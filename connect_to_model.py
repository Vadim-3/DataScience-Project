import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


def predict_image(model, image_path):
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    LABEL_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs, _ = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_name = LABEL_NAMES[predicted.item()]
        return predicted_class_name


class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        output = self.classifier(h)
        return output, h


vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = c

    return nn.Sequential(*layers)


vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
OUTPUT_DIM = 10
model = VGG(vgg11_layers, OUTPUT_DIM)
model.load_state_dict(torch.load('vgg-transfer-model.pt'))
