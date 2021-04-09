import matplotlib.pyplot as plt
import numpy as np


def plot_fig(testloader, testset):
    classes = testset.classes
    H = 10
    W = 10
    fig = plt.figure(figsize=(H, W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.4, wspace=0.4)

    for i, (images, labels) in enumerate(testloader, 0):
        for batchN in range(0, images.size()[0]):
            # numpyに変換後、[3, 32, 32] -> [32, 32, 3] に変換
            numpy_array = images[batchN].numpy().transpose((1, 2, 0))
            plt.subplot(H, W, batchN+1)
            plt.imshow(numpy_array)
            plt.title(f"{classes[labels[batchN]]}:{labels[batchN]}", fontsize=12, color = "green")
            plt.axis('off')
        break
    plt.show()
    