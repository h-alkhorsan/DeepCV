import matplotlib.pyplot as plt
import numpy as np 
import torch 

from torchvision.transforms import transforms
from torchvision import datasets 
from models import ResNet34, VGG11, GoogLeNet, InceptionV3

def imshow(image, one_channel=False):
    if one_channel:
        image = image.mean(dim=0)
    
    image = image / 2 + 0.5
    image = image.cpu().numpy()
    if one_channel:
        plt.imshow(image, cmap='Greys')
    else:
        plt.imshow(np.transpose(image, (1, 2, 0)))

def model_switcher(model, dataset):
    if model == 'vgg11':
        model = VGG11(dataset)

    elif model == 'resnet34':
        model = ResNet34(dataset)
    
    elif model == 'googlenet':
        model = GoogLeNet(dataset)
    
    elif model == 'inception_v3':
        model = InceptionV3(dataset)

    else:
        raise ValueError('Not a valid model')

    return model


def plot_predictions(model, images, labels, classes, dataset):
    output = model(images)
    _, predictions = torch.max(output, 1)
    predictions = np.squeeze(predictions.cpu().numpy())
    fig = plt.figure()

    for idx in range(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        if dataset in ['mnist', 'fashion_mnist']:
            imshow(images[idx], one_channel=True)
        else:
            imshow(images[idx], one_channel=False)
        
        ax.set_title(f'{classes[predictions[idx]]}',
        color=('green' if predictions[idx]==labels[idx].item() else 'red'),
        fontsize=10)
    fig.tight_layout()
    return fig


def load_dataset(dataset, train=False):
    classes = dict()
    classes['mnist'] = (0,1,2,3,4,5,6,7,8,9)
    classes['fashion_mnist'] = ('top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boot')
    classes['cifar10'] = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck') 

    if dataset == 'mnist':
        return datasets.MNIST(
            root='data',
            train=train,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(227),
                transforms.Normalize((0.1307), (0.3081))
            ]),
            download=False 
        ), classes['mnist']

    elif dataset == 'fashion_mnist':
        return datasets.FashionMNIST(
            root='data',
            train=train,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(227),
                transforms.Normalize((0.1307), (0.3081))
            ]),
            download=False
        ), classes['fashion_mnist']

    elif dataset == 'cifar10':
        return datasets.CIFAR10(
            root='data',
            train=train,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
                #transforms.RandomResizedCrop(224)
                #transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            download=False 
        ), classes['cifar10']

    else:
        raise ValueError('Not a valid dataset')