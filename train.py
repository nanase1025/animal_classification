import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import ExponentialLR
from dataset import*
from model import*
from torch.optim.lr_scheduler import StepLR

# Define hyperparameters
NUM_CLASSES = 22
batch_size = 32
num_workers = 0
LR = 5e-6
epochs = 500

# Define dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)

# Work on cpu or gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



# Define my model 
model = MyResNet18().to(device)
model.train()

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)
scheduler = ExponentialLR(optimizer, gamma=0.95) 
# , weight_decay=1e-8
# Define evaluation metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
losses_val = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')

# Define cosine decay function(not sure about whether or not using it. QAQ)

# Start training
max_val_accurate = 0.0
import time
start = time.time()
for i in range(epochs):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(i + 1, train_loss))
    # if i > 5:
    #     scheduler.step()
    # validate
    model.eval()    # Change to the evaluation mode
    acc = 0.0  # accumulate accurate number / epoch
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = loss_fn(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(val_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}, LR: {:.8f}'.format(i + 1, val_loss, acc, optimizer.state_dict()['param_groups'][0]['lr']))
    
    if acc >= max_val_accurate:
        torch.save(model.state_dict(), 'best_model_weights.pth')
        max_val_accurate = acc
        print("The best model has been saved~~~~~")

