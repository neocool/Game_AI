from classes.network_class import Feed_Forward
from classes.agent_class import Agent
from classes.game_class import Game
import time

def reward(agent, observation,actions):
    reward = 0
    didgood = False
    if agent.id == 1:
        didgood = True
    if didgood:
        reward +=1

    agent.score += reward


def main(agentCount):

    # Create a list of random agents 
    agents = []
    for i in range(agentCount):
        agents.append(Agent(i))
        agents[i].load_model()

    game = Game(1)
    
    #Main Loop
    
    for agent in agents:
        input("Reset game and then hit enter to start loop")
        for i in range(30):
            gameState = game.getState()    
            action = agent.decideAction(gameState)

            if action != [-1,-1]:
                game.takeAction(action)

            time.sleep(0.1)
        agent.score += int(input("Please rate the agent with a score between -10 and +10"))

    scores = []
    for agent in agents:
        scores.append(agent.score)

    index = max(enumerate(scores),key=lambda x: x[1])[0]  
    agents[index].train(1000)
    agents[index].id = 0
    agents[index].save_model()
            


main(agentCount=10)
