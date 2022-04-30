import torch
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from repvgg import create_RepVGG_A0


def matrix():

    emo_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
    deploy_model = create_RepVGG_A0(deploy=True)
    deploy_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./utils/deploy_model_30.pth').items()})
    deploy_model.cuda()

    transforms_my = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.RandomRotation(15)])

    dataset_val = ImageFolder('Y:\\RAF_Data\\val', transform=transforms_my)
    batch_size = 32
    C2 = np.zeros((7, 7))
    dataloder_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
    for i, (data, label) in enumerate(dataloder_val):

        data, label = data.cuda(), label.cuda()
        output = deploy_model(data)
        deploy_model.eval()
        pre_label = torch.argmax(output, 1)
        # print(label)
        C2 += confusion_matrix(np.array(label.cpu()), np.array(pre_label.cpu()), labels=[0, 1, 2, 3, 4, 5, 6])
        # print(pre_label)
        # print('暂停使用')

    print(C2)
    sum = np.sum(C2, axis=1)
    sum = sum.reshape(7, 1)
    sns.set()
    _, ax = plt.subplots()
    sns.heatmap(C2 / sum, annot=True, cmap='Blues', xticklabels=emo_labels, yticklabels=emo_labels)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


def open_matrix():

    path = './utils/confusion_matrix_30.png'
    img = cv2.imread(path)
    cv2.imshow("confusion_matrix", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    open_matrix()
    # matrix()


