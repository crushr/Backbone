from __future__ import print_function, division

import torch
import time
import copy
from dataloader import dataloaders
from dataloader import dataset_sizes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range (num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('——' * 20)

        # 每个epoch的训练、验证[阶段]
        for phase in ['train', 'val']:

            # 训练
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corr = 0

            #迭代
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 选择阶段
                with torch.set_grad_enabled(phase == 'train'):  # 训练时固定模型，禁止梯度传播，默认false
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # _:占位符
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)  # inputs.size(0)就是batch_size 将loss视作batch中每一个样本的loss，所以算总loss需要倍乘
                running_corr += torch.sum(preds == labels.data)  # 一个batch中的正确个数

            if phase == 'train':
                scheduler.step()  # 每个epoch学习率调整
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corr.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 验证
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    # 训练时长
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #加载最佳权重，返回模型
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, criterion, phase = 'test'):
    print('testing model')
    test_loss = 0.0
    test_corr = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_loss = criterion(outputs, labels)
            test_corr += torch.sum(preds == labels.data)
    
    test_loss /= dataset_sizes[phase]
    test_corr = test_corr.double()/dataset_sizes[phase]
    print(f"Test Erorr: \n Acc:{(100*test_corr):>0.1f}%, Avg loss: {test_loss:>8f} \n")