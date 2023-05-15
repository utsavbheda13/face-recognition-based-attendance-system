from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training will be on GPU.")
else:
    device = torch.device("cpu")
    print("Training will be on CPU.")

batch_size = 32

def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_dir = 'dataset_detected'
    dataset = datasets.ImageFolder(data_dir, transform)
    mapping = dataset.class_to_idx
    num_classes = {'num_classes': len(dataset.classes)}
    idx_to_class = {v: k for k, v in mapping.items()}
    with open('n_classes.pickle', 'wb') as file:
        pickle.dump(num_classes, file)
    with open('index_to_name.pickle', 'wb') as file:
        pickle.dump(idx_to_class, file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader, num_classes

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

def intialize_model(num_classes):
    resnet = InceptionResnetV1(pretrained='vggface2', classify=True)
    resnet = nn.Sequential(*list(resnet.children())[:-5])
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    resnet.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=128, bias=False),
        normalize()
    )
    resnet.logits = nn.Linear(128, num_classes['num_classes'])
    resnet.softmax = nn.Softmax(dim=1)
    return resnet

def training(model, loader, epochs=2, lr=0.0003, optimizer="adam"):

    epoch_tr_loss = []
    epoch_tr_acc = []

    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        total_train = 0
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_losses.append(loss.item())
            accuracy = (torch.max(outputs.data, 1)[1] == labels).sum().item()
            train_acc += accuracy
            total_train += labels.size(0)
            optimizer.step()
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total_train
        epoch_tr_loss.append(epoch_train_loss)
        epoch_tr_acc.append(epoch_train_acc)
    print(f'train_loss : {epoch_tr_loss[0]}')
    print(f'train_accuracy : {epoch_tr_acc[0]}')

def train():
    train_loader, num_classes = load_data()
    resnet = intialize_model(num_classes).to(device)
    training(resnet, train_loader)
    torch.save(resnet.state_dict(), "resnet_model.pt")