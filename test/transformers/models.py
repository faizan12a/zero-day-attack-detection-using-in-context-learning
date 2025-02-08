import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class LinearizationLayer_Soft_Labels(nn.Module):
    def __init__(self,NO_WC,NOS,Classes,NOF,device):
        super(LinearizationLayer_Soft_Labels, self).__init__()
        self.linear = nn.Linear(NO_WC*Classes, NOF) # Linearizes 1 sample's 2d matrix representing all soft labels from all weak classifiers
        self.NOS = NOS+1 # Number of samples
        self.NOF = NOF # Number of features
        self.NO_WC = NO_WC # Number of samples
        self.Classes = Classes # Number of Classes
        self.device = device
    
    def forward(self, x):
        batch = x.shape[0]
        new_x = torch.empty(x.shape[0],self.NOS,self.NOF)
        new_x = new_x.to(self.device)
        reshaped_input2 = x.reshape(batch,self.NOS,self.NO_WC*self.Classes)
        for i in range(self.NOS):
            # x.view(batch_size, -1)
            temp = reshaped_input2[:,i,:]
            new_x[:,i] = self.linear(temp)
        return new_x.to(self.device)


EMBED_DIM = 768 # Dimension of transformer input # 768 for GPT2
class DTF(torch.nn.Module):
    def __init__(self,INPUT_DIM,Classes,NOS,NO_WC,device):
        super(DTF, self).__init__()
        self.nos = NOS+1
        self.classes = Classes
        self.input_dim = INPUT_DIM
        self.flatten_linearize = LinearizationLayer_Soft_Labels(NO_WC,NOS,Classes,INPUT_DIM,device)
        self.input_proj = nn.Linear(self.input_dim, EMBED_DIM)
        self.config = AutoConfig.from_pretrained('gpt2')
        self.base_model = AutoModel.from_config(self.config)
        self.output_proj = nn.Linear(EMBED_DIM,Classes)
        self.device = device
    def forward(self, x_features,x_prob):
        x_prob = self.flatten_linearize(x_prob)
        x = torch.empty(x_features.shape[0], 2 * self.nos, self.input_dim)
        x=x.to(self.device)
        x[:, 0::2, :] = x_features
        x[:, 1::2, :] = x_prob
        
        x = self.input_proj(x)
        x = self.base_model(inputs_embeds=x).last_hidden_state
        x = self.output_proj(x)
        x = x[:,1::2]
        return x
    
class TF(torch.nn.Module):
    def __init__(self,INPUT_DIM,Classes):
        super(TF, self).__init__()
        self.input_dim = INPUT_DIM
        self.input_proj = nn.Linear(self.input_dim, EMBED_DIM)
        # self.config = AutoConfig.from_pretrained('bert-base-cased')
        self.config = AutoConfig.from_pretrained('gpt2')
        self.base_model = AutoModel.from_config(self.config)
        self.output_proj = nn.Linear(EMBED_DIM,Classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.base_model(inputs_embeds=x).last_hidden_state
        x = self.output_proj(x)
        x = x[:,1::2]
        return x