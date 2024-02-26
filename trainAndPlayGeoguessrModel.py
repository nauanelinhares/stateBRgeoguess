import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import keyboard
import mouse
import base64
from io import BytesIO

class Navegador(object):
    def __init__(self) -> None:
        self.options = self.configs()
        self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=self.options, port=9515)

    def configs(self):
        """
        Configuring ChromeDriver options
        """
        options = webdriver.ChromeOptions()
        options.add_argument('user-data-dir=C:\\Users\\Nauvo\\AppData\\Local\\Google\\Chrome\\User Data Teste\\')
        options.add_argument("--profile-directory=Profile 1")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        return options
        
    def getSite(self, site):
        """
        Open the specified site in the browser
        """
        self.driver.get(site)
    
    def getKeyPress(self):
        """
        Get the key press from the canvas element and save it as an image
        """
        # get canvas element
        # element = self.driver.find_element(By.XPATH, '//*[@id="play-component"]/div[3]/div[2]/div[3]/div/div/div[2]/div[1]/div[9]/div/div/canvas[1]')
        element = self.driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/main/div/div/div[1]/div/div/div/div/div[2]/div[1]/div[9]/div/div/canvas')
        
        canvas_base64 = self.driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", element)

        # Decode base64 image to bytes
        canvas_bytes = base64.b64decode(canvas_base64)

        # Create an image using PIL
        image = Image.open(BytesIO(canvas_bytes))

        # Save or display the image as needed
        image.save("imagem.png")

    def closeBrowser(self):
        """
        Close the browser
        """
        self.driver.quit()
        
        


class GeoguessrModel(object):
    def __init__(self, train_size, val_size, numClasses, device) -> None:
        self.device = device
        self.numClasses = numClasses
        self.train_size = train_size
        self.val_size = val_size  
        self.model = self.selectModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
    def getDataSetAndLoaderForTrain(self):
        """
        Create the dataset and data loaders for training
        """
        self.dataset = self.createDataset()
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        
    def createDataset(self):
        """
        Create the dataset
        """
        # Define dataset
        dataset = torchvision.datasets.ImageFolder(root='dataSet',transform=self.transform())
        return dataset
    
    def selectModel(self):
        """
        Select the model architecture and modify the last fully connected layer
        """
        # Use RESNET50 model
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.numClasses)
        model.to(self.device)
        return model
    
    def transform(self):
        """
        Define transformations for training
        """
        # Define transformations for training
        transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform_train
    
    def train(self):
        """
        Train the model
        """
        num_epochs = 15
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                global_step += 1
                
            # Evaluate the model on the validation set
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            # Log the accuracy for TensorBoard
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {100 * accuracy:.2f}%')
            
        # Save the model
        torch.save(self.model.state_dict(), 'model.pth')
        
    def loadModel(self):
        """
        Load the trained model
        """
        # Load the trained model file model2.pth
        self.model = torchvision.models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 27)
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.to(self.device)
        
    def getImageTransformed(self, image):
        """
        Transform the input image for prediction
        """
        transform_train = self.transform()
        image = image.convert('RGB')
        image = transform_train(image)
        image = image.to(self.device)
        return image
    
    def predictState(self, image):
        """
        Predict the state from the input image
        """
        image = self.getImageTransformed(image)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
            outputs = F.softmax(outputs, dim=1)[0]
        return outputs 
        
            

# Train with GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create GeoguessrModel object
geoguessrModel = GeoguessrModel(0.8, 0.2, 27, device)

# Array with states 
target = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
          'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 
          'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']


train = False

if train:
    geoguessrModel.getDataSetAndLoaderForTrain()
    geoguessrModel.train()

else:
    geoguessrModel.loadModel()
    geoguessr = Navegador()
    geoguessr.getSite('https://geotastic.net/')
    probsPredictState = np.ones(27)
    startPlay = False
    while True:
        
        try:
            # "Here you will log in to the site"
            if keyboard.is_pressed('esc'): 
                break
            
            if keyboard.is_pressed('ç'):
                sleep(0.05)
                # Toggle the value of startPlay
                startPlay = not startPlay
                
                if startPlay:
                    print('Start Play', 'On')
                else:
                    print('Start Play', 'Off')
                    
                probsPredictState = np.ones(27)    
                    
            # If left mouse button is clicked
            if mouse.is_pressed(button='left') and startPlay:
                geoguessr.getKeyPress()
                sleep(0.05)
                outputs = geoguessrModel.predictState(Image.open('imagem.png'))
                
                # Get probabilities and normalize
                probs = outputs.cpu().numpy()
                probs = probs / probs.sum()
                probsPredictState = probsPredictState * (probs + 0.1)
                # Normalize probsPredictState
                probsPredictState = np.array(probsPredictState)
                probsPredictState = probsPredictState / probsPredictState.sum(axis=0)
                # Clean display
                # Print State / Probability in descending order
                os.system('cls')
                for i in np.argsort(probsPredictState)[::-1]:
                    print(target[i], probsPredictState[i])
                    
                if probsPredictState.max() > 0.999:
                    # Print the state with the highest probability
                    os.system('cls')
                    print('Best Result')
                    print(target[np.argmax(probsPredictState)], probsPredictState.max())
                    startPlay = not startPlay

        except:
            print('Disable the game by pressing ç')
