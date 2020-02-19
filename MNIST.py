from torch.utils.data import Dataset, DataLoader

class MnistData (Dataset):
    def __init__(self,data,label):
        self.images = data
        self.labels = label
    def __getitem__(self,index):
        return self.images[index],self.labels[index]
    def __len__(self):
        return len(self.images)

