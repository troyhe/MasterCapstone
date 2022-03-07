import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import time
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import remove_stopwords
import gensim.downloader
import re
from torchsummary import summary

# specific package for visualization
!pip install livelossplot --quiet
from livelossplot import PlotLosses

# get the device type of machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def count_parameters(model):
  """Function for count model's parameters"""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tokenize(text):
  tokens = []
  meta = []
  for account in text.splitlines():
    account_meta, account_text = tuple(account.split(':::', 1))

    # preprocess text
    # erase URL
    account_text = re.sub(r"http\S+", "", account_text)
    # convert to lowercase
    account_text = account_text.lower()
    # remove punctuation
    account_text = strip_punctuation(account_text)
    # tokenize and remove stop word
    tokens.append(remove_stopwords(account_text).split())

    #preprocess metadata
    meta.append(list(map(int, re.findall(r'\d+', account_meta))))
  
  return tokens, meta

# read files
with open('./dataset-light-3000/train/bot.txt', 'r') as file:
  train_bot_text = file.read()
with open('./dataset-light-3000/train/human.txt', 'r') as file:
  train_human_text = file.read()
with open('./dataset-light-3000/test/bot.txt', 'r') as file:
  val_bot_text = file.read()
with open('./dataset-light-3000/test/human.txt', 'r') as file:
  val_human_text = file.read()

# tokenize text data
train_bot_tokens, train_bot_meta = tokenize(train_bot_text)
train_human_tokens, train_human_meta = tokenize(train_human_text)
val_bot_tokens, val_bot_meta = tokenize(val_bot_text)
val_human_tokens, val_human_meta = tokenize(val_human_text)

train_tokens = train_bot_tokens + train_human_tokens
train_meta = train_bot_meta + train_human_meta
val_tokens = val_bot_tokens + val_human_tokens
val_meta = val_bot_meta + val_human_meta

print(len(train_tokens))
print(len(train_meta))
print(len(val_tokens))
print(len(val_meta))

# generate ground truth
train_labels = np.concatenate((np.ones(len(train_bot_tokens)), np.zeros(len(train_human_tokens))))
val_labels = np.concatenate((np.ones(len(val_bot_tokens)), np.zeros(len(val_human_tokens))))

MAX_TWEETS = max([len(account_tokens) for account_tokens in train_tokens])
glove_twitter_embed = gensim.downloader.load('glove-twitter-25')

vector_size = 25


vector_size = 100
token_to_ix = {}
token_to_ix["\0"] = 0

for account_tokens in train_tokens:
  for token in account_tokens:
    if token not in token_to_ix:  # token has not been assigned an index yet
      token_to_ix[token] = len(token_to_ix)  # Assign each token with a unique index

len(token_to_ix)

train_vectors = np.zeros((len(train_tokens), 2000))
for i in range(len(train_tokens)):
  for j in range(min(len(train_tokens[i]), 2000)):
    train_vectors[i,j] = token_to_ix[train_tokens[i][j]]
  
val_vectors = np.zeros((len(val_tokens), 2000))
for i in range(len(val_tokens)):
  for j in range(min(len(val_tokens[i]), 2000)):
    if val_tokens[i][j] in token_to_ix:
      val_vectors[i,j] = token_to_ix[val_tokens[i][j]]



#train_set = np.stack([[glove_twitter_embed.wv[token] for token in account_tokens if token in glove_twitter_embed.wv] for account_tokens in train_tokens])
train_vectors = np.zeros((len(train_tokens), 2000, vector_size))
for i in range(len(train_tokens)):
  for j in range(min(len(train_tokens[i]), 2000)):
    if train_tokens[i][j] in glove_twitter_embed.wv:
      train_vectors[i,j] = glove_twitter_embed.wv[train_tokens[i][j]]

val_vectors = np.zeros((len(val_tokens), 2000, vector_size))
for i in range(len(val_tokens)):
  for j in range(min(len(val_tokens[i]), 2000)):
    if val_tokens[i][j] in glove_twitter_embed.wv:
      val_vectors[i,j] = glove_twitter_embed.wv[val_tokens[i][j]]

train_m_vector = np.array(train_meta)
val_m_vector = np.array(val_meta)

# randomize 
shuffler = np.random.permutation(len(train_vectors))
train_vectors = train_vectors[shuffler]
train_labels = train_labels[shuffler]
train_m_vector = train_m_vector[shuffler]



# print the shape of np array
print(train_vectors.shape)
print(train_labels.shape)
print(train_m_vector.shape)
print(val_vectors.shape)
print(val_labels.shape)
print(val_m_vector.shape)


# copy numpy data to tensor for orignal data
X_train_valid_tensor = torch.from_numpy(train_vectors).float().to(device)
#X_train_valid_tensor = torch.from_numpy(train_vectors).float().long().to(device)
y_train_valid_tensor = torch.from_numpy(train_labels).float().long().to(device) # do not forget .long()
meta_train_valid_tensor = torch.from_numpy(train_m_vector).float().to(device)

X_test_tensor = torch.from_numpy(val_vectors).float().to(device)
#X_test_tensor = torch.from_numpy(val_vectors).float().long().to(device)
y_test_tensor = torch.from_numpy(val_labels).float().long().to(device) # do not forget .long()
meta_test_tensor = torch.from_numpy(val_m_vector).float().to(device)

print ('Training/Valid tensor shape: {}'.format(X_train_valid_tensor.shape))
print ('Training/Valid target tensor shape: {}'.format(y_train_valid_tensor.shape))
print ('Training/Valid meta tensor shape: {}'.format(meta_train_valid_tensor.shape))

print ('Test tensor shape: {}'.format(X_test_tensor.shape))
print ('Test target tensor shape: {}'.format(y_test_tensor.shape))
print ('Test meta tensor shape: {}'.format(meta_test_tensor.shape))


class EmbDataset(Dataset):
  """EEG dataset."""
  def __init__(self, subset, transform=None):
    self.subset = subset
    self.transform = transform
        
  def __getitem__(self, index):
    x, y, meta = self.subset[index]
    if self.transform:
      pass 
      # x = self.transform(x)
      # y = self.transform(y)
    return x, y, meta
        
  def __len__(self):
    return len(self.subset)

# create dataloader for orignal data
init_dataset = TensorDataset(X_train_valid_tensor, y_train_valid_tensor, meta_train_valid_tensor) 
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, meta_test_tensor)

# split train and val
lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)] 
subset_train, subset_val = random_split(init_dataset, lengths) 

train_data = EmbDataset(subset_train, transform=None)

val_data = EmbDataset(subset_val, transform=None)

# create dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)
}
 
test_data = EmbDataset(test_dataset, transform=None)


def train_model(model, optimizer, num_epochs, dataloaders):
    # for each epoch... 
    liveloss = PlotLosses()

    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      logs = {}

      # let every epoch go through one training cycle and one validation cycle
      # TRAINING AND THEN VALIDATION LOOP...
      for phase in ['train', 'val']:
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        start_time = time.time()
        # first loop is training, second loop through is validation
        # this conditional section picks out either a train mode or validation mode
        # depending on where we are in the overall training process
        # SELECT PROPER MODE- train or val
        if phase == 'train':
          for param_group in optimizer.param_groups:
            print("LR", param_group['lr']) # print out the learning rate
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode
        
        for inputs, labels, metas in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)
          metas = metas.to(device)
          batch_idx += 1
          
          optimizer.zero_grad()
          
          with torch.set_grad_enabled(phase == 'train'):
          #    the above line says to disable gradient tracking for validation
          #    which makes sense since the model is in evluation mode and we 
          #    don't want to track gradients for validation)
            outputs = model(inputs, metas)
            # compute loss where the loss function will be defined later
            
            loss = loss_fn(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()
            train_loss += loss
            _, predicted = outputs.max(1)
            #predicted = (outputs > 0.5).long()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


        prefix = ''
        cur_loss = train_loss.item()/(batch_idx)

        if phase == 'val':
            prefix = 'val_'

            if cur_loss < best_valid_loss:
                best_valid_loss = cur_loss
                torch.save(model.state_dict(), 'best_model.pt')
        
        logs[prefix + 'loss'] = cur_loss
        logs[prefix + 'acc'] = correct/total*100.

      liveloss.update(logs)
      liveloss.send()

    # end of single epoch iteration... repeat of n epochs  
    return model

def test_model(model,test_data,criterion):
    
    
    # Creating the test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=0)
    
    # Making the predictions on the dataset
    
    total_test_preds = 0
    correct_test_preds = 0
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        
        for test_inputs, test_labels, test_metas in test_dataloader:
            
            # Transfer test data and labels to device
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_metas = test_metas.to(device)
            
            # Perform forward pass
            test_outputs = model(test_inputs, test_metas)
            
            # Compute loss
            test_loss = criterion(test_outputs,test_labels)
            
            # Compute test statistics
                    
            test_loss += test_loss.item()
            _, test_predicted = test_outputs.max(1)
            #test_predicted = (test_outputs > 0.5).long()
            total_test_preds += test_labels.size(0)
            correct_test_preds += test_predicted.eq(test_labels).sum().item()
            
        test_acc = correct_test_preds/total_test_preds
        print('Test loss', test_loss)
        print('Test accuracy',test_acc*100)
        
    
    return test_acc



#LSTM with pretrained embedding


class LSTMModel(nn.Module):

  def __init__(self, input_dim, hidden_dim1, hidden_dim2, layer_dim, output_dim, dropout=0.0):
    super(LSTMModel, self).__init__()

    # Hidden dimensions
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    # Number of hidden layers
    self.layer_dim = layer_dim

    # LSTM layer
    self.lstm = nn.LSTM(input_dim, hidden_dim1, layer_dim, dropout=dropout, batch_first=True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, input_dim)
    # output layer
    self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc2 = nn.Linear(hidden_dim2, output_dim)
    
  def forward(self, x): # (B, 2000, 25)
    x, _ = self.lstm(x, self.init_hidden(x.size(0))) # (B, 2000, 50)
    x = self.fc1(x[:, -1, :]) # (B, 25)
    x = self.fc2(x) # (B, 2)
    return x
  
  def init_hidden(self, batch_size):
    return (torch.zeros(self.layer_dim, batch_size, self.hidden_dim1).to(device), torch.zeros(self.layer_dim, batch_size, self.hidden_dim1).to(device))

class LSTMModel_v3(nn.Module):

  def __init__(self, input_dim, hidden_dim, layer_dim, dropout=0.0):
    super(LSTMModel_v3, self).__init__()

    # Hidden dimensions
    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    # LSTM layer
    self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout, batch_first=True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, input_dim)
    # output layer
    self.fc = nn.Linear(hidden_dim, 2)
    
  def forward(self, x): # (B, 2000, 25)
    x, _ = self.lstm(x, self.init_hidden(x.size(0))) # (B, 2000, 50)
    x = self.fc(x[:, -1, :]) # (B, 2)
    return x
  
  def init_hidden(self, batch_size):
    return (torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device), torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device))


class LSTMModel_v2(nn.Module):

  def __init__(self, embedding_dim, vocab_size, hidden_dim1, hidden_dim2, layer_dim, output_dim, dropout=0.0):
    super(LSTMModel_v2, self).__init__()

    # Hidden dimensions
    self.embedding_dim = embedding_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.vocab_size = vocab_size
    # Number of hidden layers
    self.layer_dim = layer_dim

    # embedding layer
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    # LSTM layer
    self.lstm = nn.LSTM(embedding_dim, hidden_dim1, layer_dim, dropout=dropout, batch_first=True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, input_dim)
    # output layer
    self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc2 = nn.Linear(hidden_dim2, output_dim)
    
  def forward(self, x): # (B, 2000)
    x = self.word_embeddings(x) # (B, 2000, 100)
    x, _ = self.lstm(x, self.init_hidden(x.size(0))) # (B, 2000, 32)
    x = self.fc1(x[:, -1, :]) # (B, 16)
    x = self.fc2(x) # (B, 2)
    return x
  
  def init_hidden(self, batch_size):
    return (torch.zeros(self.layer_dim, batch_size, self.hidden_dim1).to(device), torch.zeros(self.layer_dim, batch_size, self.hidden_dim1).to(device))



class CNN_v1(nn.Module):
  def __init__(self, in_channels, classes):
    super(CNN_v1, self).__init__()

    # Define the first conv layer
    self.conv1 = nn.Conv2d(in_channels, 8, (21, 4))
    # define the ELU activation layer
    self.elu = nn.ELU()
    # define the first batch normalization layer
    self.bn1 = nn.BatchNorm2d(8)
    # define the seoncd conv layer
    self.conv2 = nn.Conv2d(8, 4, (32, 9))
    # define the second batch normalization layer
    self.bn2 = nn.BatchNorm2d(4)
    # define the maxpool layer
    self.maxpool = nn.MaxPool2d((4,1))
    # define the dropout layer
    self.dropout = nn.Dropout(0.5)

    # linear layers
    self.fc1 = nn.Linear(116, 4)
    #self.fc2 = nn.Linear(228, classes)
    self.fc2 = nn.Linear(228, 36)
    self.fc3 = nn.Linear(36, classes)

  def forward(self, x, z): # (B, 2000, 25), (B, 4)
    # pipeline for x
    x = x.view(-1, 1, 2000, 25) # (B, 1, 2000, 25)

    x = self.conv1(x) # (B, 8, 1980, 20)
    x = self.elu(x) # (B, 8, 1980, 20)
    x = self.bn1(x) # (B, 8, 1980, 20)
    x = self.maxpool(x) # (B, 8, 495, 20)

    x = self.conv2(x) # (B, 4, 464, 14) 
    x = self.elu(x) # (B, 4, 464, 14)
    x = self.bn2(x) # (B, 4, 464, 14) 
    x = self.maxpool(x) # (B, 4, 116, 14)
    x = self.dropout(x) # (B, 4, 116, 14) added

    x = x.permute(0, 1, 3, 2) # # (B, 4, 14, 116)
    x = x.reshape(-1, 56, 116) # (B, 56, 116)
    x = self.fc1(x) # (B, 56, 4)
    x = x.view(-1, 56*4) # (B, 224)
    x = torch.cat((x, z), dim=1) # (B, 228)
    x = self.elu(x) # (B, 228)
    x = self.fc2(x) # (B, 32)
    x = self.elu(x)
    x = self.fc3(x) # (B, 2)
    return x

class CNN_v2(nn.Module):
  def __init__(self, in_channels, classes):
    super(CNN_v2, self).__init__()

    # Define the first conv layer
    self.conv1 = nn.Conv2d(in_channels, 16, (21, 4))
    # define the ELU activation layer
    self.elu = nn.ELU()
    # define the first batch normalization layer
    self.bn1 = nn.BatchNorm2d(16)
    # define the seoncd conv layer
    self.conv2 = nn.Conv2d(16, 8, (32, 9))
    # define the second batch normalization layer
    self.bn2 = nn.BatchNorm2d(8)
    # define the third conv layer
    self.conv3 = nn.Conv2d(8, 4, (25, 7))
    # define the thord batch normalization layer
    self.bn3 = nn.BatchNorm2d(4)
    # define the maxpool layer
    self.maxpool = nn.MaxPool2d((4,1))


    # define the dropout layer
    self.dropout = nn.Dropout(0.5)

    # linear layer
    self.fc1 = nn.Linear(23, 4)
    self.fc2 = nn.Linear(128, classes)

  def forward(self, x): # (B, 2000, 25)
    x = x.view(-1, 1, 2000, 25) # (B, 1, 2000, 25)

    x = self.conv1(x) # (B, 16, 1980, 22)
    x = self.elu(x) # (B, 16, 1980, 22)
    x = self.bn1(x) # (B, 16, 1980, 22)
    x = self.maxpool(x) # (B, 16, 495, 22)

    x = self.conv2(x) # (B, 8, 464, 14)
    x = self.elu(x) # (B, 8, 464, 14)
    x = self.bn2(x) # (B, 8, 464, 14)
    x = self.maxpool(x) # (B, 8, 116, 14)
    x = self.dropout(x) # (B, 8, 116, 14)

    x = self.conv3(x) # (B, 4, 92, 8)
    x = self.elu(x) # (B, 4, 92, 88
    x = self.bn3(x) # (B, 4, 92, 8) 
    x = self.maxpool(x) # (B, 4, 23, 8)
    x = self.dropout(x) # (B, 4, 23, 8) 
    # try to permute in the other way next time
    x = x.permute(0, 1, 3, 2) # # (B, 4, 8, 23)
    x = x.reshape(-1, 32, 23) # (B, 32, 23)
    x = self.fc1(x) # (B, 32, 4)
    x = x.view(-1, 128) # (B, 128)
    x = self.fc2(x) # (B, 2)
    return x


class CRNN_v1(nn.Module):
  def __init__(self, in_channels, classes):
    super(CRNN_v1, self).__init__()

    # Define the first conv layer
    self.conv1 = nn.Conv2d(in_channels, 8, (21, 4))
    # define the ELU activation layer
    self.elu = nn.ELU()
    # define the first batch normalization layer
    self.bn1 = nn.BatchNorm2d(8)
    # define the seoncd conv layer
    self.conv2 = nn.Conv2d(8, 4, (32, 9))
    # define the second batch normalization layer
    self.bn2 = nn.BatchNorm2d(4)
    # define the maxpool layer
    self.maxpool = nn.MaxPool2d((4,1))
    # define the dropout layer
    self.dropout = nn.Dropout(0.5)

    # define the third GRU layer
    self.gru = nn.GRU(1, 16, 2, batch_first=True, bidirectional=True)

    # define linear layers
    self.fc1 = nn.Linear(116, 4)
    self.fc2 = nn.Linear(32, classes)
  
  def init_hidden(self, batch_size, hidden_dim, layer_dim = 1):
    return torch.zeros(layer_dim*2, batch_size, hidden_dim).to(device)

  def forward(self, x): # (B, 2000, 25)
    x = x.view(-1, 1, 2000, 25) # (B, 1, 2000, 25)

    x = self.conv1(x) # (B, 8, 1980, 20)
    x = self.elu(x) # (B, 8, 1980, 20)
    x = self.bn1(x) # (B, 8, 1980, 20)
    x = self.maxpool(x) # (B, 8, 495, 20)

    x = self.conv2(x) # (B, 4, 464, 14) 
    x = self.elu(x) # (B, 4, 464, 14)
    x = self.bn2(x) # (B, 4, 464, 14) 
    x = self.maxpool(x) # (B, 4, 116, 14)
    x = self.dropout(x) # (B, 4, 116, 14) added

    x = x.permute(0, 1, 3, 2) # # (B, 4, 14, 116)
    x = x.reshape(-1, 56, 116) # (B, 56, 116)
    x = self.fc1(x) # (B, 56, 4)

    x = x.view(-1, 56*4, 1) # (B, 56*4)
    x, _ = self.gru(x, self.init_hidden(x.shape[0],16,2)) # (B, 56*4, 32)
    x = self.dropout(x)

    x = self.fc2(x[:,-1,:]) # (B, 2)
    return x


# define the hyperparamters
weight_decay = 0.001  # weight decay to alleviate overfiting

model = LSTMModel(input_dim=25, hidden_dim1=32, hidden_dim2=16, layer_dim=1, output_dim=2).to(device)

count = count_parameters(model)
print ('model parameters amount {}'.format(count))

#loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5, weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay=weight_decay)


model=train_model(model, optimizer, num_epochs=50, dataloaders=dataloaders)



# define the hyperparamters
weight_decay = 0.001  # weight decay to alleviate overfiting

model = LSTMModel_v2(embedding_dim=100, vocab_size=len(token_to_ix), hidden_dim1=32, hidden_dim2=16, layer_dim=1, output_dim=2).to(device)

count = count_parameters(model)
print ('model parameters amount {}'.format(count))

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5, weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr = 3e-4, weight_decay=weight_decay)


model=train_model(model, optimizer, num_epochs=100, dataloaders=dataloaders)

weight_decay = 0.001  # weight decay to alleviate overfiting

model = LSTMModel_v3(input_dim=25, hidden_dim=32, layer_dim=1).to(device)

count = count_parameters(model)
print ('model parameters amount {}'.format(count))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay=weight_decay)


model=train_model(model, optimizer, num_epochs=10, dataloaders=dataloaders)



# define the hyperparamters
#weight_decay = 0.007 # weight decay to alleviate overfiting
weight_decay = 0.018 # weight decay to alleviate overfiting

model4 = CNN_v1(in_channels=1, classes=2).to(device)

count = count_parameters(model4)
print ('model parameters amount {}'.format(count))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model4.parameters(), lr = 1e-4, weight_decay=weight_decay)

model4=train_model(model4, optimizer, num_epochs=70, dataloaders=dataloaders)

test_result = test_model(model4,test_data,loss_fn)

torch.save(model4, "CNN_m_v2.pt")

summary(model4, [(2000, 25), (4,)])


# define the hyperparamters
#weight_decay = 0.007 # weight decay to alleviate overfiting
weight_decay = 0.01 # weight decay to alleviate overfiting

model5 = CNN_v2(in_channels=1, classes=2).to(device)

count = count_parameters(model5)
print ('model parameters amount {}'.format(count))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model5.parameters(), lr = 4e-5, weight_decay=weight_decay)
model5=train_model(model5, optimizer, num_epochs=50, dataloaders=dataloaders)
test_result = test_model(model5,test_data,loss_fn)
torch.save(model5, "CNN_v2.pt")

weight_decay = 0.01 # weight decay to alleviate overfiting

model6 = CRNN_v1(in_channels=1, classes=2).to(device)

count = count_parameters(model6)
print ('model parameters amount {}'.format(count))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model6.parameters(), lr = 1e-4, weight_decay=weight_decay)

model6=train_model(model6, optimizer, num_epochs=40, dataloaders=dataloaders)

test_result = test_model(model6,test_data,loss_fn)

torch.save(model6, "CNN_v1-2.pt")