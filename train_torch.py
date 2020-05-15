import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pickle

def load_file(file):
    with open(file,'rb') as f:
        return pickle.load(f)

class CNN_LSTM(nn.Module):
    def __init__(self,seqlen,embedding_matrix,emb_size,kernel,filters,neurons,drop):
        super(CNN_LSTM, self).__init__()
        self.model = nn.Sequential()
        '''Input: (*)(∗) , LongTensor of arbitrary shape containing the indices to extract
           Output: (*, H)(∗,H) , where * is the input shape and H=embedding_dim'''
        self.model.add_module('emb',nn.Embedding.from_pretrained(embeddings=embedding_matrix,freeze=True))#freeze=True词向量训练过程中不改变
        self.model.add_module('conv1',nn.Conv1d(in_channels=emb_size, out_channels=filters, kernel_size=kernel))
        self.model.add_module('maxpool', nn.MaxPool1d(kernel_size=2, stride=1)) #out_size=filters-1
        self.model.add_module('lstm',nn.LSTM(input_size=filters-1, hidden_size=neurons, batch_first=True))
        self.model.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.model.add_module('drop',nn.Dropout(p=drop))
        self.model.add_module('dense',nn.Linear(in_features=seqlen*neurons, out_features=3, bias=True))
        self.model.add_module('softmax',nn.Softmax())
    def forward(self, x):
        z = self.model(x)
        return z

class MyDataset(Dataset):
    def __init__(self, transform=None, x_path=None, y_path=None):
        #self.transform = transform
        self.x_path = x_path
        self.y_path = y_path
        #self.labels = pd.read_csv(self.label_path)
        self.x = load_file(self.x_path)
        self.y = load_file(self.y_path)

    def __getitem__(self, index):
        return self.x[index], self.y[index]  # 返回一个元组

    def __len__(self):
        return len(self.x)

train_data = MyDataset(x_path='./x_train.pkl',y_path='./y_train.pkl')
valid_data = MyDataset(x_path='./x_valid.pkl',y_path='./y_valid.pkl')

train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True,drop_last=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=128, shuffle=True,drop_last=True)

embedding_matrix = load_file('./embedding_matrix.pkl')
seqlen = 100
emb_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kernel = 3
filters=32
neurons = 64
drop = 0.5
model = CNN_LSTM(seqlen,embedding_matrix,emb_size,kernel,filters,neurons,drop)

criterion = nn.CrossEntropyLoss() #损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) #优化器
model.to(device) #gpu/cpu

epochs = 1 #200*8 ?
steps = 0
running_loss = 0
train_loss = [] #用于后面画图

for epoch in range(epochs):
    for inputs, labels in train_loader:  # batch128,
        print("Steps: ", steps)
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)  # 模型预测值
     #   print(type(logps))
      #  print(logps)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()