import torch
import matplotlib.pyplot as plt
import numpy as np

def miscImages(model, device, test_loader, classes):
    model.eval()
    test_loss = 0
    incorrect = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            for i in range(len(target)):
              if pred[i].item() != target[i]:
                incorrect += 1
                print('\n\n{} [ Predicted Value: {}, Actual Value: {} ]'.format(
                incorrect, classes[pred[i].item()], classes[target[i]], ))
                print(data[i].cpu().numpy().shape)
                #dataa = np.rollaxis(data[i].cpu().numpy(),0,3)
                #
                dataa = data[i].cpu()
                img = np.transpose(dataa, (1, 2, 0))
                print(img.shape)
                # img[0] = 
                # img[1] =
                # img[2] = 
                plt.imshow(img/2 + 0.5)
                plt.show()