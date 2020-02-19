# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torch.nn as torchNN
import numpy as np
from model import LeNet
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as PPLT
from MNIST import MnistData
import matplotlib.image as mpimg
import gzip
import os
import argparse
import random

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(480, 40, 211, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(470, 110, 211, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(470, 200, 211, 61))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(500, 270, 181, 61))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(480, 360, 181, 111))
        self.pushButton_5.setObjectName("pushButton_5")
        self.textEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(200, 350, 201, 91))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.button1)
        self.pushButton_2.clicked.connect(self.button2)
        self.pushButton_3.clicked.connect(self.button3)
        self.pushButton_4.clicked.connect(self.button4)
        self.pushButton_5.clicked.connect(self.button5)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "5.1 show train images"))
        self.pushButton_2.setText(_translate("MainWindow", "5.2 show hyperparameters"))
        self.pushButton_3.setText(_translate("MainWindow", "5.3 train 1 epoch"))
        self.pushButton_4.setText(_translate("MainWindow", "5.4 show trainning result"))
        self.pushButton_5.setText(_translate("MainWindow", "5.5 inference "))
    def button1(self):
        showImage()
    def button2(self):
        printParameter()
    def button3(self):
        trainModel(EPOCH_NUM=1,show=True)
    def button4(self):
        PPLT.figure()
        img = mpimg.imread('image.jpg')
        PPLT.imshow(img)
        PPLT.show()
    def button5(self):
        index =  self.textEdit.text()
        Inference(int(index))

class LeNet(torchNN.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = torchNN.Sequential(
            torchNN.Conv2d(1,6,5,1,2),
            torchNN.ReLU(),
            torchNN.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2 = torchNN.Sequential(
            torchNN.Conv2d(6,16,5),
            torchNN.ReLU(),
            torchNN.MaxPool2d(2,2)
        )
        self.fc1 = torchNN.Sequential(
            torchNN.Linear(16 *5 *5,120),
            torchNN.ReLU()
        )
        self.fc2 = torchNN.Sequential(
            torchNN.Linear(120,84),
            torchNN.ReLU()
        )
        self.fc3 = torchNN.Sequential(
            torchNN.Linear(84,10),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def get_data(image_url,label_url):
    images = gzip.GzipFile(image_url,'rb').read()
    labels = gzip.GzipFile(label_url,'rb').read()
    img_num = int.from_bytes(images[4:8],byteorder='big')
    label_num = int.from_bytes(labels[4:8], byteorder='big')
    assert (img_num == label_num)
    row = int.from_bytes(images[8:12],byteorder ='big')
    col = int.from_bytes(images[12:16],byteorder='big')
    img_size = row * col
    x,y=[],[]
    for i in range(img_num):
        img_offset = 16 + img_size *i
        lbl_offest = 8 + i
        img = torch.Tensor(list(images[img_offset:img_offset+img_size])).float()
        img = img.view(1,row,col)
        lbl = int(labels[lbl_offest])
        x.append(img)
        y.append(lbl)
    return x,y

def trainModel(EPOCH_NUM=50,save=False,show=False):
    net = LeNet().to(devices)
    criterion = torchNN.CrossEntropyLoss()
    optimizer =torch.optim.SGD(net.parameters(),lr=LR)
    batch,batchloss =[],[]
    for epoch in range(EPOCH_NUM):
        sum_loss = 0.0
        acc = 0
        iter = 0
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(devices), labels.to(devices)
            # forward and backward
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _ ,pred = torch.max(outputs.data,1)
            acc += (pred == labels).sum()
            iter = iter +1
            batch.append(i)
            batchloss.append(loss.item())
        if show == True :
            PPLT.figure()
            PPLT.plot(batch, batchloss, 'b')
            PPLT.title('one epoch')
            PPLT.xlabel('iteration')
            PPLT.ylabel('loss')
            PPLT.show()
        print('Epoch [%d] : loss [%f]'%(epoch+1,sum_loss/iter))
        print('train accuracy = %f%%'%(100*acc/len(trainData)))

root = os.getcwd()
# CPU or GPU
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#parameter
BATCH_SIZE = 32
LR = 0.001

# get training data

trainImage,trainLabel = get_data(os.path.join(root,"dataset/train-images-idx3-ubyte.gz"),os.path.join(root,"dataset/train-labels-idx1-ubyte.gz"))
trainData = MnistData(trainImage,trainLabel)
trainLoader = DataLoader(dataset=trainData,
                         batch_size=BATCH_SIZE,
                         shuffle=True )
# get testing data
testImage,testLabel = get_data(os.path.join(root,"dataset/t10k-images-idx3-ubyte.gz"),os.path.join(root,"dataset/t10k-labels-idx1-ubyte.gz"))
testData = MnistData(testImage,testLabel)
testLoader = DataLoader(dataset=testData,
                         batch_size=BATCH_SIZE,
                         shuffle=True )

def showImage(number = 10) :
    fig,ax = PPLT.subplots(nrows=1,ncols=10,sharex='all',sharey='all')
    ax = ax.flatten()
    len =  trainData.__len__()
    for i in range(number):
        x = random.randint(0,len-1)
        img,label = trainData.__getitem__(x)
        ax[i].imshow(img.reshape(28,28),cmap='Greys',interpolation='nearest')
        ax[i].set_title(label)
    PPLT.show()

def printParameter():
    print('hyperparameters:')
    print('batch size:',BATCH_SIZE)
    print('learning rate:',LR)
    print ('optimizer:SGD')

def Inference(index):
    model = LeNet()
    model.load_state_dict(torch.load('MNIST_Model.pth'))
    img,label= testData.__getitem__(index)
    img = img.unsqueeze(0)
    output = model(img)
    _,pred = torch.max(output,1)
    output = output.tolist()
    output = [i/100 for i in output[0]]
    x = [0,1,2,3,4,5,6,7,8,9] 
    PPLT.figure()
    PPLT.subplot(2,1,1)
    PPLT.imshow(img.reshape(28,28))
    PPLT.subplot(2,1,2)
    PPLT.ylim(-1,1)
    PPLT.yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    PPLT.xlim(0,9)
    PPLT.bar(x,output)
    PPLT.show()

class MnistData (Dataset):
    def __init__(self,data,label):
        self.images = data
        self.labels = label
    def __getitem__(self,index):
        return self.images[index],self.labels[index]
    def __len__(self):
        return len(self.images)






if(__name__ == '__main__'):
     import sys
     app = QtWidgets.QApplication(sys.argv)
     MainWindow = QtWidgets.QMainWindow()
     ui = Ui_MainWindow()
     ui.setupUi(MainWindow)
     MainWindow.show()
     sys.exit(app.exec_())
