import numpy as np
import matplotlib.pyplot as plt


def plot_loss(epochs, train_loss, v_loss, title):
    plt.figure(figsize=(8,8))
    x = np.arange(1,epochs+2)
    plt.plot(x, train_loss, label = 'Training Loss')
    plt.plot(x, v_loss, label = 'Validation Loss')
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Loss', fontsize =16)
    plt.title(title,fontsize =16)
    plt.legend(fontsize=16)
    
    
def plot_acc(epochs,v_acc):
    plt.figure(figsize=(8,8))
    x = np.arange(1,epochs+2)
    plt.plot(x, v_acc)
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Validation Accuracy', fontsize =16)
    plt.title('Validation Accuracy v/s Epochs',fontsize =16)
