import numpy as np
import os
import torch
from utils.IDPdataset import SiteDataset, PairDataset
from sklearn.preprocessing import StandardScaler
from models.ETGplusLSTM_version1 import ETG_LSTM_final
from utils.analysis import analysis
scaler = StandardScaler()
import warnings
warnings.filterwarnings("ignore")
NUMBER_EPOCHS = 50

Dataset_Path = "./"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SiteDataset("./")
model2=ETG_LSTM_final().to(device)
def train_one_epoch(model, train_set):
    epoch_loss_train = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    for _, data in enumerate(train_set):
        h = None
        c = None
        for time, snapshot in enumerate(data):
            snapshot.x = torch.from_numpy(scaler.fit_transform(snapshot.x))
            snapshot.x = snapshot.x.to(torch.float32)
            snapshot = snapshot.to(device)
            x = snapshot.x[:, 3:14]
            pos = snapshot.x[:, 0:3]


            if time==15: #output in the last time step
                y_true = snapshot.y
                y_pred,c = model(x=x,
                                 pos=pos,
                                 h=h,
                                 c=c, 
                                 output= True)
            else:
                h,c = model(x=x,
                            pos=pos,
                            h=h,
                            c=c,
                            output= False)
        model.optimizer.zero_grad()
        loss = model.criterion(y_pred, y_true)
        #for analysis
        softmax = torch.nn.Softmax(dim=1)
        y_pred1 = softmax(y_pred)
        y_pred1 = y_pred1.cpu().detach().numpy()
        y_true1 = y_true.cpu().detach().numpy()
        valid_pred += [pred[1] for pred in y_pred1]
        valid_true += list(y_true1)
        # backward gradient
        loss.backward()
        model.optimizer.step()
        epoch_loss_train += loss.item()
        n += 1
    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg, valid_true, valid_pred
def train_full_model(train_set):
    print("\nTraining a full model using all training data...\n")
    model = ETG_LSTM_final().to(device)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg, train_true, train_pred = train_one_epoch(model, train_set)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

train_full_model(dataset)
#train_one_epoch(model2, dataset)