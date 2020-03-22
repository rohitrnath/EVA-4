import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

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


def torchTransforms(train = 1):
	if(train == 1):
		train_transforms = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    Cutout(8),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # RGB mean and variance
		])

	else:
	test_transforms = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

