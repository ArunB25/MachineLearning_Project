import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from torchvision import transforms
import json

class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        for param in self.model.parameters(): 
            param.requires_grad = False
        fc_inputs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential( ##Change final layer
            torch.nn.Linear(fc_inputs, 13),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)) 

    def forward(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch) # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

        output = output.squeeze()
        category_probs = {}
        for idx, probability in enumerate(output):
            category_probs.update({f"{categories[str(idx)]}":round(probability.item(),4)})

        print(category_probs)
        return category_probs


    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            pass

    def predict_classes(self, image):
        with torch.no_grad():
            pass


try:
    Image_model  = ImageClassifier()
    state_dict = torch.load("model_epoch79.pt")
    Image_model.load_state_dict(state_dict, strict= False)

    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    with open('categories_dict.json', 'r') as file:
        categories = json.load(file)
    pass
except:
    raise OSError("No categorys found. Check that you have the category file in the correct location")

app = FastAPI()
print("Starting server")

@app.get("/")
async def root():
    return {"Uvicorn": "I'm alive"}

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    predictions = Image_model(pil_image)
    
    return JSONResponse(content={
    "Predicted Category": f"{max(predictions, key=predictions.get)}", # Return the category here
    "All predictions": f"{predictions}" # Return a list or dict of probabilities here
    })
    
    
# if __name__ == '__main__':
#   uvicorn.run("api:app", host="0.0.0.0  ", port=80)