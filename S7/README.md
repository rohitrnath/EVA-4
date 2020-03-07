# EVA-4 Assignment 7

## CIFAR-10 - Object Recognition in Images

### Got 80% accuracy within 9 epochs

### Labels Test Accuracies
Accuracy of plane : 84 %
Accuracy of   car : 91 %
Accuracy of  bird : 75 %
Accuracy of   cat : 73 %
Accuracy of  deer : 82 %
Accuracy of   dog : 51 %
Accuracy of  frog : 78 %
Accuracy of horse : 78 %
Accuracy of  ship : 84 %
Accuracy of truck : 88 %

### Model Summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 28, 28]           1,728
              ReLU-2           [-1, 64, 28, 28]               0
       BatchNorm2d-3           [-1, 64, 28, 28]             128
           Dropout-4           [-1, 64, 28, 28]               0
            Conv2d-5           [-1, 64, 26, 26]          36,864
              ReLU-6           [-1, 64, 26, 26]               0
       BatchNorm2d-7           [-1, 64, 26, 26]             128
           Dropout-8           [-1, 64, 26, 26]               0
            Conv2d-9           [-1, 64, 26, 28]             256
           Conv2d-10          [-1, 128, 26, 26]          24,704
             ReLU-11          [-1, 128, 26, 26]               0
      BatchNorm2d-12          [-1, 128, 26, 26]             256
          Dropout-13          [-1, 128, 26, 26]               0
           Conv2d-14          [-1, 128, 26, 26]           1,280
           Conv2d-15          [-1, 256, 26, 26]          33,024
             ReLU-16          [-1, 256, 26, 26]               0
      BatchNorm2d-17          [-1, 256, 26, 26]             512
          Dropout-18          [-1, 256, 26, 26]               0
        MaxPool2d-19          [-1, 256, 13, 13]               0
           Conv2d-20           [-1, 64, 13, 13]          16,384
      BatchNorm2d-21           [-1, 64, 13, 13]             128
          Dropout-22           [-1, 64, 13, 13]               0
           Conv2d-23           [-1, 64, 11, 11]          36,864
             ReLU-24           [-1, 64, 11, 11]               0
      BatchNorm2d-25           [-1, 64, 11, 11]             128
          Dropout-26           [-1, 64, 11, 11]               0
           Conv2d-27           [-1, 64, 11, 11]             640
           Conv2d-28          [-1, 128, 11, 11]           8,320
             ReLU-29          [-1, 128, 11, 11]               0
      BatchNorm2d-30          [-1, 128, 11, 11]             256
          Dropout-31          [-1, 128, 11, 11]               0
           Conv2d-32          [-1, 128, 11, 11]           1,280
           Conv2d-33          [-1, 256, 11, 11]          33,024
             ReLU-34          [-1, 256, 11, 11]               0
      BatchNorm2d-35          [-1, 256, 11, 11]             512
          Dropout-36          [-1, 256, 11, 11]               0
           Conv2d-37          [-1, 256, 11, 13]           1,024
           Conv2d-38          [-1, 512, 11, 11]         393,728
        AvgPool2d-39            [-1, 512, 1, 1]               0
           Conv2d-40             [-1, 10, 1, 1]           5,120
================================================================
Total params: 596,288
Trainable params: 596,288
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 14.95
Params size (MB): 2.27
Estimated Total Size (MB): 17.24
----------------------------------------------------------------


### First 9 Epochs

0%|          | 0/391 [00:00<?, ?it/s]EPOCH: 0
Loss=1.7481470108032227 Batch_id=390 Accuracy=41.32: 100%|██████████| 391/391 [00:35<00:00, 11.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.5857, Accuracy: 4357/10000 (43.57%)

EPOCH: 1
Loss=1.5722239017486572 Batch_id=390 Accuracy=59.01: 100%|██████████| 391/391 [00:35<00:00, 11.92it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.1254, Accuracy: 6029/10000 (60.29%)

EPOCH: 2
Loss=1.572922945022583 Batch_id=390 Accuracy=65.89: 100%|██████████| 391/391 [00:35<00:00, 11.72it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.0281, Accuracy: 6255/10000 (62.55%)

EPOCH: 3
Loss=1.3345487117767334 Batch_id=390 Accuracy=69.75: 100%|██████████| 391/391 [00:35<00:00, 11.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8590, Accuracy: 6942/10000 (69.42%)

EPOCH: 4
Loss=1.4623113870620728 Batch_id=390 Accuracy=72.60: 100%|██████████| 391/391 [00:35<00:00, 11.91it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8361, Accuracy: 7026/10000 (70.26%)

EPOCH: 5
Loss=1.338636875152588 Batch_id=390 Accuracy=74.17: 100%|██████████| 391/391 [00:35<00:00, 12.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7573, Accuracy: 7373/10000 (73.73%)

EPOCH: 6
Loss=1.264021635055542 Batch_id=390 Accuracy=75.65: 100%|██████████| 391/391 [00:35<00:00, 11.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7132, Accuracy: 7504/10000 (75.04%)

EPOCH: 7
Loss=1.3522580862045288 Batch_id=390 Accuracy=76.52: 100%|██████████| 391/391 [00:35<00:00, 11.13it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7419, Accuracy: 7388/10000 (73.88%)

EPOCH: 8
Loss=1.5451769828796387 Batch_id=390 Accuracy=77.98: 100%|██████████| 391/391 [00:35<00:00, 11.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6910, Accuracy: 7600/10000 (76.00%)

EPOCH: 9
Loss=1.0949978828430176 Batch_id=390 Accuracy=81.35: 100%|██████████| 391/391 [00:35<00:00, 11.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5805, Accuracy: 8009/10000 (80.09%)
