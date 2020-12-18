
import os
import numpy as np
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from warpctc_pytorch import CTCLoss
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os
# os.chdir('/home/lab507c/chineseocr')
# os.chdir('../../')
from dataset import PathDataset,randomSequentialSampler,alignCollate
from glob import glob
from sklearn.model_selection import train_test_split

alphabetChinese = 'WH1JFj47VzuowRnx2eiD3bAvpUgZKd8fINQctGqPOsTLSEBM9lX0YhaCrkmy56'

trainP,testP = train_test_split(roots,test_size=0.1)##此处未考虑字符平衡划分
traindataset = PathDataset(trainP,alphabetChinese)
testdataset = PathDataset(testP,alphabetChinese)
print(testdataset[0])
batchSize = 100
workers = 8
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh =256
sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=None,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

train_iter = iter(train_loader)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
from crnn.util import strLabelConverter
lr = 0.1
optimizer = optim.Adadelta(model.parameters(), lr=lr)
converter = strLabelConverter(''.join(alphabetChinese))
criterion = CTCLoss()

from train.ocr.dataset import resizeNormalize
from crnn.util import loadData
image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    print("True")
    model.cuda()
#     model = torch.nn.DataParallel(model, device_ids=[1])##转换为多GPU训练模型
    image = image.cuda()
    criterion = criterion.cuda()

def trainBatch(net, criterion, optimizer,cpu_images, cpu_texts):
    #data = train_iter.next()
    #cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    
    loadData(text, t)
    loadData(length, l)
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost




def predict(im):
    """
    预测
    """
    image = im.convert('L')
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))
    
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred
   
   
def val(net, dataset, max_iter=100):

    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    N = len(dataset)
    
    max_iter = min(max_iter, N)
    for i in range(max_iter):
        im,label = dataset[np.random.randint(0,N)]
        if im.size[0]>1024:
            continue
        
        pred = predict(im)
        if pred.strip() ==label:
                n_correct += 1

    accuracy = n_correct / float(max_iter )
    return accuracy

from train.ocr.generic_utils import Progbar

nepochs = 10
acc  = 0

interval = len(train_loader)//2##评估模型
lr = 0.1
optimizer = optim.Adadelta(model.parameters(), lr=lr)
            
for i in range(0,nepochs):
    if i ==5:
        lr = 0.01
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    if i ==8:
        lr = 0.001
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    print('epoch:{}/{}'.format(i,nepochs))
    n = len(train_loader)
    pbar = Progbar(target=n)
    train_iter = iter(train_loader)
    loss = 0
    for j in range(n):
        for p in model.named_parameters():
            p[1].requires_grad = True


        model.train()
        try:
            cpu_images, cpu_texts = train_iter.next()
            cost = trainBatch(model, criterion, optimizer,cpu_images, cpu_texts)

            loss += cost.data.numpy()
            
            if (j+1)%interval==0:
                curAcc = val(model, testdataset, max_iter=1024)
                if curAcc>acc:
                    acc = curAcc
                    torch.save(model.state_dict(), 'train/ocr/modellstm.pth')
                
                
            pbar.update(j+1,values=[('loss',loss/((j+1)*batchSize)),('acc',acc)])
        except:
            print("error")