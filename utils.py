# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.
# Created by Junbo Zhao 2020/3/19
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(args, total_num=50000):
    data_dir, batch_size, num_workers, resize = args.data_dir, args.batch_size, args.num_workers, args.resize
    
    if resize:
        train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([transforms.ToTensor()])

    

    # index = np.arange(total_num)
    # np.random.shuffle(index)
    # train_index = index
    # test_index = index

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/val', transform=test_transforms)
    train_total = len(train_dataset)
    test_total = len(test_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,  num_workers=num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    return train_loader, test_loader, train_total,test_total

# Test the model on clean dataset
def test(model, dataloader):
    model.eval()

    preds = []
    with torch.no_grad():
        for (images, _) in dataloader:
            images = images.cuda()

            images = normalize(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted)
    all = torch.concat(preds)
    label = torch.mode(all)[0]

    total = all.shape[0]
    labels = label.repeat(total)
    acc = torch.sum(all == labels)/labels.shape[0]
    h, w = images.shape[2], images.shape[3]
    return acc, labels, h, w

# Load the log and generate the training line
def log_generation(exp_dir):
    log_dir = os.path.join(exp_dir,'log.csv')
    # Load the statistics in the log
    epochs, train_rate, test_rate = [], [], []
    with open(log_dir, 'r') as f:
        reader = csv.reader(f)

        for i in reader:
            epochs.append(int(i[0]))
            train_rate.append(float(i[1]))


    # Generate the success line
    plt.figure(num=0)
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    plt.savefig(os.path.join(exp_dir,'success_rate.png'))
    plt.close(0)