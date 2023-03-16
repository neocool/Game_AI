import torch
import torch.nn as nn

class Feed_Forward(nn.Module):
	def __init__(self,D_IN,D_OUT,hidden_nodes):
		super(Feed_Forward, self).__init__()
						
		#Network 1 
		self.input1 = nn.Linear(D_IN,hidden_nodes,bias=True)
		self.hidden1 = nn.Linear(hidden_nodes,hidden_nodes,bias=True)
		self.hidden2 = nn.Linear(hidden_nodes,hidden_nodes,bias=True)
		self.hidden3 = nn.Linear(hidden_nodes,hidden_nodes,bias=True)
		self.hidden4 = nn.Linear(hidden_nodes,hidden_nodes,bias=True)
		self.output1 = nn.Linear(hidden_nodes,D_OUT,bias=True)
		
	def forward(self,x):
		x = x.view(-1, 3*48*48)
		nn1_pred = self.input1(x)
		nn1_pred = self.hidden1(nn1_pred)
		nn1_pred = self.hidden2(nn1_pred)
		nn1_pred = self.hidden3(nn1_pred)
		nn1_pred = self.hidden4(nn1_pred)
		nn1_pred = self.output1(nn1_pred)
		return nn1_pred