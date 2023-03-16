from .network_class import Feed_Forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class Agent():
    def __init__(self,id):
        self.id = id 
        self.score = 0
        self.model = 0
        self.actions = []
        self.states = []

    def create_model(self):
        self.model = Feed_Forward(D_IN=6912,D_OUT=2,hidden_nodes=1000)

    def train(self,epochs,save=True):
        if self.model == 0:
            self.create_model()
        features = torch.tensor(self.states)
        labels = torch.tensor(self.actions)

        print(features.size())
        print(labels.size())
        input()
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3)       
        loss_fn = nn.L1Loss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            prediction = self.model(features)
            loss = loss_fn(prediction, labels)  
            loss.backward()
            optimizer.step()
        
        last_loss = loss.item()
        print("loss: " + str(last_loss))   

    def test(self,features,labels):
        pass
        
    def predict(self,state):
        featues =  torch.tensor([state])
        prediction = self.model(featues)
        return prediction.tolist()[0]


    def save_model(self):
        print("Saving Model")  
        torch.save(self.model,"Models/model"+str(self.id))

    def load_model(self):
        try:
            self.model = torch.load("Models/model"+str(self.id))
        except:
            pass

    def decideAction(self,gameState,variableRate=0.7):
        self.states.append(gameState)
        action_list = []
        if self.model == 0:            
            for i in range(2):
                action_list.append(random.random())
        else:
            if random.random() > variableRate:
                for i in range(2):
                    action_list.append(random.random())
            else:
                prediction = self.predict(gameState)

                if prediction[0] < 0 or prediction[0] > 1:
                    prediction[0] = random.random()
                if prediction[1] < 0 or prediction[1] > 1:
                    prediction[1] = random.random()

                action_list = prediction

        self.actions.append(action_list)

        return action_list
        

