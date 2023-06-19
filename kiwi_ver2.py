import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import sys
from customdataset_3 import CustomImageDataset
from confusion_matrix_210825 import confusion_matrix
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, latent_dim=1000, img_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(256, 512, bias=True),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(512, 1024, bias=True),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(1024, 2048, bias=True),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(2048, 4096, bias=True),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(4096, 3 * 128 * 128, bias=True),
                                   nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, 128, 128)
        return x


# Discriminator model

def runVgg16GAN(train_loader, validation_loader, train_set, validation_set):
    global d_performance, g_performance
    g_model = Generator()
    d_model = models.vgg16(weights=True)

    # num_features = model.classifier[6].in_features 의 6은 고정
    # g_features = g_model.classifier[6].in_features
    d_features = d_model.classifier[6].in_features
    # for classifier in model.classifier :
    #    print(classifier)

    # print(list(g_model.classifier.children())[-1])
    # g_features = list(g_model.classifier.children())
    # g_model.classifier = nn.Sequential(*g_features)

    d_features = list(d_model.classifier.children())
    d_features.extend([nn.Linear(1000, 1), nn.Sigmoid()])
    d_model.classifier = nn.Sequential(*d_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = g_model.to(device)
    d = d_model.to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(g.parameters(), lr=0.0001)
    d_optimizer = optim.SGD(d.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=50, eta_min=0)

    num_epoch = 200
    learning_rate = 0.0002
    img_size = 224 * 224
    num_channel = 1
    batch_size = 64

    min_loss = 10

    for epoch in range(300):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0
        total_correct = 0

        total_loss = 0
        total_correct = 0

        for i, data in loop:
            g_optimizer.zero_grad()

            real_images = data['image'].to(device)

            batch_size = real_images.size(dim=0)
            real_label = torch.full(size=[batch_size, 1], fill_value=1, dtype=torch.float32).to(device)
            fake_label = torch.full(size=[batch_size, 1], fill_value=0, dtype=torch.float32).to(device)


            z = torch.randn(batch_size, 256).to(device)

            outputs = g(z)

            #print(outputs.shape)

            g_loss = criterion(d(outputs), real_label)

            #print(g_loss.item())

            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()

            z = torch.randn(batch_size, 256).to(device)

            fake_images = g(z)

            # Calculate fake & real loss with generated images above & real images
            fake_loss = criterion(d(fake_images), fake_label)
            real_loss = criterion(d(real_images), real_label)
            d_loss = (fake_loss + real_loss) / 2

            #print("Dis")

           # print(d_loss.item())

            # Train discriminator with backpropagation
            # In this part, we don't train generator
            d_loss.backward()
            d_optimizer.step()

        z = torch.randn(batch_size, 256).to(device)

        fake_images = g(z)

        save_image(fake_images, os.path.join('GAN_fake_samples.png'))


def main():
    trans_train = transforms.Compose([transforms.Resize((128, 128)),  # 224 x 224 size 로 resize
                                      # (200, 200)은 출력할 size를 조정
                                      # scale(0.1, 1)은 면적의 비율 0.1~1 (10%~100%)를 무작위로 자르기
                                      # ratio(0.5, 2)은 면적의 너비와 높이의 비율 0.5~2를 무작위로 조절
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_validation = transforms.Compose([transforms.Resize((64, 64)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_set = CustomImageDataset(
        data_set_path="C:/Users/LEESaebom/PycharmProjects/kiwi/dataset/train",
        transforms=trans_train)
    val_data_set = CustomImageDataset(
        data_set_path="C:/Users/LEESaebom/PycharmProjects/kiwi/dataset/train",
        transforms=trans_train)

    # train_loader = DataLoader(train_data_set, num_workers=2, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data_set, num_workers=6, batch_size=16, shuffle=True)

    # runVgg16(train_loader, val_loader, train_data_set, val_data_set)
    # runResnet(train_loader, val_loader, train_data_set, val_data_set)
    # runDensenet(train_loader, val_loader, train_data_set, val_data_set)
    runVgg16GAN(val_loader, val_loader, val_data_set, val_data_set)
    # runVIT(train_loader, val_loader, train_data_set, val_data_set)


if __name__ == '__main__':
    main()
