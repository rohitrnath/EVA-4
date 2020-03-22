import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from AlbumentationTransform import *

# preprocessing

class Cutout(object):
  def __init__(self, sz):
    self._sz = sz

  def __call__(self, img):
    h = img.size(1)
    w = img.size(2)

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = int(np.clip(y - self._sz / 2, 0, h))
    y2 = int(np.clip(y + self._sz / 2, 0, h))
    x1 = int(np.clip(x - self._sz / 2, 0, w))
    x2 = int(np.clip(x + self._sz / 2, 0, w))
    img[:, y1:y2, x1:x2].fill_(0.0)
    return img

def pullCIFAR10(train_transforms, test_transforms):

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=train_transforms)


	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=test_transforms)

	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
	    torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader,  testloader, classes


def analyseCIFAR10():
	# simple transform
	simple_transforms = transforms.Compose([
	                                      #  transforms.Resize((28, 28)),
	                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
	                                       transforms.ToTensor(),
	                                      #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
	                                       # Note the difference between (0.1307) and (0.1307,)
	                                       ])
	exp = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=simple_transforms)
	exp_data = exp.data
	print(' - Numpy Shape:', exp.data.shape)
	#print(' - Tensor Shape:', exp.data.size())
	exp_data1 = exp.transform(exp_data[:,:,:,0])
	print('[CHANNEL 0]')
	#print(' - Numpy Shape:', exp.data.cpu().numpy().shape)
	#print(' - Tensor Shape:', exp.data.size())
	print(' - min:', torch.min(exp_data1))
	print(' - max:', torch.max(exp_data1))
	print(' - mean:', torch.mean(exp_data1))
	print(' - std:', torch.std(exp_data1))
	print(' - var:', torch.var(exp_data1))

	exp_data2 = exp.transform(exp_data[:,:,:,1])
	print('[CHANNEL 1]')
	#print(' - Numpy Shape:', exp.data.cpu().numpy().shape)
	#print(' - Tensor Shape:', exp.data.size())
	print(' - min:', torch.min(exp_data2))
	print(' - max:', torch.max(exp_data2))
	print(' - mean:', torch.mean(exp_data2))
	print(' - std:', torch.std(exp_data2))
	print(' - var:', torch.var(exp_data2))

	exp_data3 = exp.transform(exp_data[:,:,:,2])
	print('[CHANNEL 2]')
	#print(' - Numpy Shape:', exp.data.cpu().numpy().shape)
	#print(' - Tensor Shape:', exp.data.size())
	print(' - min:', torch.min(exp_data3))
	print(' - max:', torch.max(exp_data3))
	print(' - mean:', torch.mean(exp_data3))
	print(' - std:', torch.std(exp_data3))
	print(' - var:', torch.var(exp_data3))

