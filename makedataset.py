from utils.IDPdataset import SiteDataset, PairDataset
import torch
dataset = SiteDataset("./")
#print(dataset[1].features.shape)

for _,data in enumerate(dataset):
    for time, snapshot in enumerate(data):
        print(time, snapshot.x)

'''
loaddata = torch.load('./processed/data_a0.pt')
for time, snapshot in enumerate(loaddata):
    print(time, snapshot)
'''