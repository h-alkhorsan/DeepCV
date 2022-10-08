import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import plot_predictions, load_dataset, model_switcher
from datetime import datetime 

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--dataset', default=None, type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model_switcher(args.model, args.dataset)
model = model.to(device)

log_path = f'experiments/{model.name}/{args.dataset}'

if not os.path.exists(log_path):
    os.makedirs(log_path)

timestamp = datetime.now().strftime('%I.%M.%S.%d.%m.%y')
writer = SummaryWriter(f'{log_path}/exp_{timestamp}')

dataset_train, classes = load_dataset(args.dataset, train=True)
dataset_test, classes = load_dataset(args.dataset, train=False)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

num_epochs = 25
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if model.name == 'googlenet': # adapt to googlenet
            aux1, aux2, output = model(images)
            loss = criterion(output, labels)
            loss_aux1 = criterion(aux1, labels)            
            loss_aux2 = criterion(aux2, labels)            
            loss = loss + 0.3 * (loss_aux1 + loss_aux2)

        elif model.name == 'inception_v3': # adapt to inception_v3
            aux, output = model(images)
            loss = criterion(output, labels)
            loss_aux = criterion(aux, labels)
            loss = loss + 0.3 * loss_aux 

        else:
            output = model(images)
            loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        _, predictions = torch.max(output, 1)

        total += labels.size(0)
        correct += (predictions==labels).sum().item()

        running_loss += loss.item()

     
        if (batch_idx + 1) % 100 == 0:
            print(f'[Train] Epoch: [{epoch}/{num_epochs}], Step: [{batch_idx * len(images)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)], Loss: [{loss.item():.4f}]')
            logfile = open(f'{log_path}/exp_{timestamp}/log.txt', 'a')
            print(f'[Train] Epoch: [{epoch}/{num_epochs}], Step: [{batch_idx * len(images)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)], Loss: [{loss.item():.4f}]', file=logfile)
            logfile.close()

    running_loss /= len(train_loader)
    accuracy = 100. * correct / total

    writer.add_scalar('loss/train', running_loss, epoch)
    writer.add_scalar('accuracy/train', accuracy, epoch)
    writer.flush()

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            
            _, predictions = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predictions==labels).sum().item()

            running_loss += loss.item()

    running_loss /= len(test_loader)  
    accuracy = 100. * correct / total

    print(f'\n[Test] Loss: [{running_loss:.4f}], Accuracy: [{correct}/{total} ({accuracy:.0f}%)]\n')
    logfile = open(f'{log_path}/exp_{timestamp}/log.txt', 'a')
    print(f'\n[Test] Loss: [{running_loss:.4f}], Accuracy: [{correct}/{total} ({accuracy:.0f}%)]\n', file=logfile)
    logfile.close()
                
    writer.add_scalar('loss/test', running_loss, epoch)
    writer.add_scalar('accuracy/test', accuracy, epoch)
    writer.add_figure('predictions', plot_predictions(model, images, labels, classes, args.dataset), global_step=epoch)
    writer.flush()
    plt.close('all')  

if __name__ == '__main__':
    for epoch in range(1, num_epochs+1):
        train(epoch)
        test(epoch)
    print('Model training completed...')

    save_path = f'weights/{model.name}/{args.dataset}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), f'{save_path}/exp_{timestamp}.pth')




