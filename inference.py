import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
import os
from PIL import Image
from torch.utils.data import Dataset
import json

LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat",
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish",
    12: "horse", 13: "lion", 14: "lobster",
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake",
    19: "spider", 20: "turkey", 21: "wolf"
}

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_files = self._get_image_files()

    def __len__(self):
        return len(self.image_files)

    def _get_image_files(self):
        image_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:  
                filename = file.split(".")[0]
                image_files.append(os.path.join(root, file))
                
        image_files.sort(key=lambda x: x[0])

        return image_files
    
    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


test_data_path = "your_test_data_path" # you should modify this path
output_file = "output.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model parameters
model_path = "/root/code/homework1/best_model_weights.pth"
my_model = MyResNet18().to(device)  
my_model.load_state_dict(torch.load(model_path))
my_model.eval()

# transform the data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load test dataset
test_dataset = CustomDataset(test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# output
results = []
print("Test start")
for images in test_loader:
    images = images.to(device)
    with torch.no_grad():
        outputs = my_model(images)
        predicted = torch.argmax(outputs.data, 1)
        predicted_index = predicted.item()
        predicted_label = LABEL_MAP[predicted_index]
        results.append(predicted_label)

# Save output to JSON file
output_dict = {"predictions": results}

new_output = "{"
for i, prediction in enumerate(output_dict["predictions"][:-1]):
    new_output += f' "{i}": "{prediction}",\n'
i, prediction = list(enumerate(output_dict["predictions"]))[-1]
new_output += f' "{i}": "{prediction}"'

new_output += "\n}"

with open("output.json", "w") as file:
    file.write(new_output)
