import numpy as np
import itertools as it
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Creating dataset
exp_matrix = np.loadtxt('seqfish/count_matrix/OB_expression.txt', usecols=range(1,2051),skiprows=1)
print(exp_matrix.shape)
coords = np.loadtxt('seqfish/cell_locations/OB_centroids_coord.txt',skiprows=1, usecols=(1,2))
channel_info = np.loadtxt('seqfish/cell_locations/OB_centroids_annot.txt',skiprows=1, usecols=(1),dtype=int)

class GeneExpressionData(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None):
    super(GeneExpressionData, self).__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return []

  @property
  def processed_file_names(self):
    return ['data.pt']

  def download(self):
    pass

  def process(self):
    # Read data into huge `Data` list.
    data_list = []
    dict= {'source':[],'target':[],'distance':[]}

    #creating edge list for each plate, no connections between nodes of 1 plate with other
    # total number of nodes will be total number of cells across all plates = 2050
    for plate in range(7):
      indices = np.where(channel_info==plate)
      # for each plate get all possible pairs of cells and calculate distance
      all_pairs = list(it.combinations(np.unique(indices),2))
      sources= [i[0] for i in all_pairs]
      targets = [i[1] for i in all_pairs]
      # euclidean distance between pairs of cells will be edge attributes
      distances = np.linalg.norm(np.take(coords,sources,axis=0)-np.take(coords,targets,axis=0),axis=1).tolist()
      dict['source'].extend(sources)
      dict['target'].extend(targets)
      dict['distance'].extend(distances)
      assert(len(dict['source'])==len(dict['target'])==len(dict['distance']))

    edge_index = torch.tensor([dict['source'],
                                   dict['target']], dtype=torch.long).cuda()
    edge_attr = torch.tensor(dict['distance'], dtype=torch.long).cuda()
    index_list = np.arange(2050)
    for row in range(exp_matrix.shape[0]):
      x = torch.tensor(exp_matrix[row,:], dtype=torch.float)
      data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
      data_list.append(data)

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2050, 128) # num_of_feature,embed_dim
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index,edge_attr)

        return F.log_softmax(x, dim=1)

def train_sdcn(train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # # TODO:
        loss = 0
        loss.backward()
        optimizer.step()

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = GeneExpressionData(root='../').cuda()
print(dataset.shape)

dataset = dataset.shuffle()
train_dataset = dataset[:8000]
val_dataset = dataset[8000:]
print(train_dataset.shape,val_dataset.shape)

batch_size= 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
