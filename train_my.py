import torch
import h5py
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from repvgg import create_RepVGG_A0

train_model = create_RepVGG_A0(deploy=False)
train_model[0].load_state_dict(torch.load('./utils/RepVGG-A0-train.pth'))
train_model.cuda()

transforms_my = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.RandomRotation(15)])

dataset_train = ImageFolder('Y:\\RAF_Data\\train', transform=transforms_my)
dataset_val = ImageFolder('Y:\\RAF_Data\\val', transform=transforms_my)
batch_size = 32
batch_num_train = len(dataset_train.targets) // batch_size
batch_num_val = len(dataset_val.targets) // batch_size
dataloder_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dataloder_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(train_model.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-8)
loss_func = nn.CrossEntropyLoss()
train_loss, val_loss = [], []
train_acc, val_acc = [], []
epochs = 30
for epoch in range(epochs):

    train_loss_epoch, val_loss_epoch = 0, 0
    train_corrects, val_corrects = 0, 0
    print('开始第{0}次迭代'.format(epoch + 1))
    for i, (data, label) in enumerate(dataloder_train):

        data, label = data.cuda(), label.cuda()
        output = train_model(data)
        loss = loss_func(output, label)
        pre_label = torch.argmax(output, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * data.size(0)
        train_corrects += torch.sum(pre_label == label.data)
        print('already train{0} / {1} examples'.format(i+1, batch_num_train))

    print('第{0}epoch的train_loss为{1}'.format(epoch+1, train_loss_epoch / len(dataset_train.targets)))
    print('第{0}epoch的train_acc为{1}'.format(epoch+1, train_corrects / len(dataset_train.targets)))
    train_loss.append(train_loss_epoch / len(dataset_train.targets))
    train_acc.append(train_corrects / len(dataset_train.targets))

    train_model.eval()
    for j, (data_val, label_val) in enumerate(dataloder_val):

        data_val, label_val = data_val.cuda(), label_val.cuda()
        output = train_model(data_val)
        loss = loss_func(output, label_val)
        pre_label = torch.argmax(output, 1)
        val_loss_epoch += loss.item() * data_val.size(0)
        val_corrects += torch.sum(pre_label == label_val.data)
        print('already val{0} / {1} examples'.format(j+1, batch_num_val))

    print('第{0}epoch的val_loss为{1}'.format(epoch + 1, val_loss_epoch / len(dataset_val.targets)))
    print('第{0}epoch的val_acc为{1}'.format(epoch + 1, val_corrects / len(dataset_val.targets)))
    val_loss.append(val_loss_epoch / len(dataset_val.targets))
    val_acc.append(val_corrects / len(dataset_val.targets))

    if (epoch + 1) % 10 == 0:
        torch.save(train_model.state_dict(), './utils/train_model_{0}'.format(epoch+1))

file = h5py.File('./utils/info.h5', 'w')
file['train_loss'] = torch.tensor(train_loss)
file['train_acc'] = torch.tensor(train_acc)
file['val_loss'] = torch.tensor(val_loss)
file['val_acc'] = torch.tensor(val_acc)
file.close()
