from PIL import ImageGrab
from PIL import Image
import pyautogui

from torchvision import transforms
p = transforms.Compose([transforms.Resize((48,48))])

import torchvision.transforms as transforms
transform = transforms.ToTensor()


class Game():
    def __init__(self,id):
        self.id = id
        self.screenWidth, self.screenHeight = pyautogui.size()

    def getState(self):
        # Define a region of interest on the screen (top, left, width, height)
        top, left, width, height = 0 , 0, self.screenWidth , self.screenHeight
        region = (top, left, width, height)
        # Take a screenshot of that region
        screenshot = ImageGrab.grab(bbox=region)
        screenshot = p(screenshot)
    
        # Convert the screenshot to a tensor        
        tensor = transform(screenshot)
        state = tensor.tolist()
        return state

    def takeAction(self,action):
        pyautogui.moveTo(self.screenWidth * action[0], self.screenHeight *  action[1])
        pyautogui.click()