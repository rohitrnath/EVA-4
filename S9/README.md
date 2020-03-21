# EVA-4 Assignment 8

## CIFAR-10 - Object Recognition with RESNET18

### Got 80% accuracy within 9 epochs

Accuracy of plane : 88 %
Accuracy of   car : 100 %
Accuracy of  bird : 78 %
Accuracy of   cat : 73 %
Accuracy of  deer : 90 %
Accuracy of   dog : 80 %
Accuracy of  frog : 90 %
Accuracy of horse : 91 %
Accuracy of  ship : 100 %
Accuracy of truck : 91 %

### Model Summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------

### Training Logs

0%|          | 0/391 [00:00<?, ?it/s]EPOCH: 0
Loss=1.3924500942230225 Batch_id=390 Accuracy=41.62: 100%|██████████| 391/391 [00:53<00:00,  7.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -3.8742, Accuracy: 5242/10000 (52.42%)

EPOCH: 1
Loss=0.9219139814376831 Batch_id=390 Accuracy=59.75: 100%|██████████| 391/391 [00:53<00:00,  7.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.1465, Accuracy: 6388/10000 (63.88%)

EPOCH: 2
Loss=0.9044997096061707 Batch_id=390 Accuracy=68.46: 100%|██████████| 391/391 [00:54<00:00,  7.23it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.5648, Accuracy: 7478/10000 (74.78%)

EPOCH: 3
Loss=0.547200083732605 Batch_id=390 Accuracy=74.50: 100%|██████████| 391/391 [00:54<00:00,  7.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.5997, Accuracy: 7542/10000 (75.42%)

EPOCH: 4
Loss=0.9648700952529907 Batch_id=390 Accuracy=77.64: 100%|██████████| 391/391 [00:54<00:00,  7.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.0490, Accuracy: 7660/10000 (76.60%)

EPOCH: 5
Loss=0.43833741545677185 Batch_id=390 Accuracy=80.30: 100%|██████████| 391/391 [00:54<00:00,  7.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.4047, Accuracy: 7833/10000 (78.33%)

EPOCH: 6
Loss=0.4944685101509094 Batch_id=390 Accuracy=82.16: 100%|██████████| 391/391 [00:54<00:00,  7.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -9.2730, Accuracy: 8150/10000 (81.50%)

EPOCH: 7
Loss=0.3833875060081482 Batch_id=390 Accuracy=83.87: 100%|██████████| 391/391 [00:55<00:00,  7.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -10.0269, Accuracy: 8326/10000 (83.26%)

EPOCH: 8
Loss=0.28818219900131226 Batch_id=390 Accuracy=84.95: 100%|██████████| 391/391 [00:55<00:00,  7.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -10.3782, Accuracy: 8582/10000 (85.82%)

EPOCH: 9
Loss=0.2086803913116455 Batch_id=390 Accuracy=88.77: 100%|██████████| 391/391 [00:55<00:00,  7.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -11.0678, Accuracy: 8837/10000 (88.37%)

EPOCH: 10
Loss=0.5027881860733032 Batch_id=390 Accuracy=89.65: 100%|██████████| 391/391 [00:56<00:00,  6.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -11.5250, Accuracy: 8889/10000 (88.89%)

EPOCH: 11
Loss=0.19587399065494537 Batch_id=390 Accuracy=90.19: 100%|██████████| 391/391 [00:56<00:00,  6.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -11.5428, Accuracy: 8895/10000 (88.95%)

EPOCH: 12
Loss=0.19153572618961334 Batch_id=390 Accuracy=90.45: 100%|██████████| 391/391 [00:56<00:00,  6.94it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -11.6604, Accuracy: 8926/10000 (89.26%)

EPOCH: 13
Loss=0.18481048941612244 Batch_id=390 Accuracy=90.87: 100%|██████████| 391/391 [00:56<00:00,  6.94it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.0822, Accuracy: 8925/10000 (89.25%)

EPOCH: 14
Loss=0.1992826759815216 Batch_id=390 Accuracy=91.18: 100%|██████████| 391/391 [00:56<00:00,  6.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.1223, Accuracy: 8959/10000 (89.59%)

EPOCH: 15
Loss=0.22543497383594513 Batch_id=390 Accuracy=91.36: 100%|██████████| 391/391 [00:56<00:00,  6.91it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.4441, Accuracy: 8957/10000 (89.57%)

EPOCH: 16
Loss=0.1263706386089325 Batch_id=390 Accuracy=91.74: 100%|██████████| 391/391 [00:56<00:00,  6.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.4518, Accuracy: 8977/10000 (89.77%)

EPOCH: 17
Loss=0.2596966028213501 Batch_id=390 Accuracy=91.89: 100%|██████████| 391/391 [00:56<00:00,  6.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.4092, Accuracy: 8972/10000 (89.72%)

EPOCH: 18
Loss=0.22064344584941864 Batch_id=390 Accuracy=92.24: 100%|██████████| 391/391 [00:56<00:00,  6.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -12.5453, Accuracy: 9003/10000 (90.03%)

EPOCH: 19
Loss=0.20484885573387146 Batch_id=390 Accuracy=92.53: 100%|██████████| 391/391 [00:55<00:00,  7.00it/s]

Test set: Average loss: -12.5902, Accuracy: 9002/10000 (90.02%)




