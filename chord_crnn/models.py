import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import ceil

def calculate_padding(input_dim, output_dim, kernel_size, stride, dilation=1):
    padding = ((output_dim - 1) * stride + (kernel_size - 1) * (dilation - 1) + kernel_size - input_dim) / 2
    padding = ceil(padding)
    # padding = (kernel_size - 1) / dilation
    # print('padding: ', padding)
    return int(padding)

class Artist(nn.Module):
    def __init__(self):
        super(Artist, self).__init__()

    def forward(self, x):
        out = x
        return out


class Genre(nn.Module):
    def __init__(self):
        super(Genre, self).__init__()

    def forward(self, x):
        out = x
        return out


class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()
        self.fc0a = nn.Sequential(
            nn.Linear(14,144),
            nn.ReLU()
        )
        self.layer1a = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=(4), stride=2, padding=(0)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2)))
        self.layer1b = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=(4), stride=2, padding=(0)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2)))
        self.layer2a = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(1), stride=1, padding=(0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
        self.layer2b = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(1), stride=1, padding=(0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
        self.layer3a = nn.Sequential(
            #nn.Conv1d(128, 256, kernel_size=(1), stride=1, padding=(0)),
            #nn.Conv1d(256,256,kernel_size=(1),stride=1, padding=(0)),
            nn.Linear(256,256),
            nn.ReLU())
            #nn.AdaptiveMaxPool1d(1))
        # self.layer3b = nn.Sequential(
        #     nn.Conv1d(128, 256, kernel_size=(1), stride=1, padding=(0)),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool1d(1))
        
        self.hidden_size = 256
        self.input_size = 128
        self.i2h = nn.GRU(self.input_size, self.hidden_size,batch_first=True,num_layers=2)
        
        self.stepdown_layer1 = nn.Linear(self.hidden_size,27)

        
        
    def forward(self, x_a,x_b):
        batch_size = x_b.shape[0]
        x_a = x_a.view(batch_size,-1,12)
        x_b = x_b.view(batch_size,-1,2)
        x_in = torch.cat((x_a,x_b),2)
        x_in = x_in.view(batch_size,-1,14)
        #x_a = self.fc0a(x_in)
        #print("in ",x_b.shape)
        #print("x_a ",x_a[0][0])
        #print("x_b ",x_b)
        #x_a = torch.DoubleTensor(x_a[0])
        #print("x_a after ",x_a.shape)
        x_a = x_a.view(batch_size,1,-1)
        #print("in a ",x_a.shape," in b ",x_b.shape)
        #print("x_a ",x_a.size())
        out_a = self.layer1a(x_a)
        #print("1a: ",out_a.shape)
        #x_b = x_b.view(batch_size,1,-1)
        #print("x_b ",x_b.shape)
        #out_b = self.layer1b(x_b)
        #print('1b: ', out_b.shape)

        #print("out ",out.shape)
        out_a = self.layer2a(out_a)
        #print("out 2a ",out_a.shape)
        #out_b = self.layer2b(out_b)
        #print("ob ",out_b.shape)
        #out_a = out_a.view(batch_size,-1,1)
        #out_b = out_b.view(batch_size,-1,1)
        #out_b = self.layer3b(out_b)
        #out = torch.cat((out_a, out_b), 1)
        #print("o size ",out.shape)
        #out = out.view(batch_size,-1,128)
        #print("out shape ",out.shape)
        #out_a = self.layer3a(out)
        #print("out shape 2 ",out_a.shape)
        #print("concatenated ",out.shape)
        out_a = out_a.view(batch_size,128,-1)
        #self.i2h.input_size = (out.shape[1] * out.shape[2])
        #print('hs ',self.input_size)
        #out = out.view(batch_size,-1,self.i2h.input_size)
        #print("o shape ",out.shape)
        #out = torch.add(out_a,1,torch.mean(out_b))
        out = out_a.transpose(1,2)
        #print("trans ",out.shape)
        out, self.hidden = self.i2h(out, self.hidden)
        #print('2:',out.shape)
        #out = out.view(self.hidden_size, -1)
        out = self.stepdown_layer1(out)
        
        #print('stl ', out.shape)
        #out = out.reshape(batch_size,1,-1,128)
        #print("out ",out.shape)
        #out = self.output_layer(out)
        #print("out ",out.shape)
        #print("befor softmax ",out)
        #out = self.lin2(out)
        #out = self.lin3(out)
        #print('out last',out.shape)
        #out = self.lin3(out)
        
        
        return out
    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        self.hidden = Variable(torch.zeros(2,mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
        #######################################
        ### END OF YOUR CODE
        #######################################



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #############################################################################
        # TODO: Implement the ConvNet as per the instructions given in the assignment
        # 3 layer CNN followed by 1 fully connected layer. Use only ReLU activation
        #   CNN #1: k=8, f=3, p=1, s=1
        #   Max Pool #1: pooling size=2, s=2
        #   CNN #2: k=16, f=3, p=1, s=1
        #   Max Pool #2: pooling size=4, s=4
        #   CNN #3: k=32, f=3, p=1, s=1
        #   Max Pool #3: pooling size=4, s=4
        #   FC #1: 64 hidden units                                                             
        ############################################################################# 
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))    
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc1_relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for the model
        ############################################################################# 
        #print('x shape: ',x.shape)
        out = self.layer1(x)
        #print('1: ', out.shape)
        out = self.layer2(out)
        #print('2:',out.shape)
        out = self.layer3(out)
        out = out.view(list(x.shape)[0],-1)
        #print('lin_in: ', out.shape)
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.fc2(out)
        #print('out: ', out.shape)
        return out
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Implement your own model based on the hyperparameters of your choice
        ############################################################################# 
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))    
        self.fc1 = torch.nn.Linear(2048, 64)
        self.fc1_relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for your model
        ############################################################################# 
        #print('x shape: ',x.shape)
        out = self.layer1(x)
        #print('1: ', out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        #print('2:',out.shape)
        out = out.view(list(x.shape)[0],-1)
        #print('lin_in: ', out.shape)
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.fc2(out)
        #print('out: ', out.shape)
        return out

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################