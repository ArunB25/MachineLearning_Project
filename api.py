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


class TextClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(TextClassifier, self).__init__()
        pass

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the text model    #
##############################################################
        
        self.decoder = decoder
    def forward(self, text):
        x = self.main(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            pass


    def predict_classes(self, text):
        with torch.no_grad():
            pass

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

        # probabilities = torch.nn.functional.softmax(output[0], dim=0)# The output has unnormalized scores. To get probabilities, you can run a softmax on it.

        # # Show top categories per image
        # top5_prob, top5_catid = torch.topk(probabilities, 5)
        # top_categories = {}
        # for i in range(top5_prob.size(0)):
        #     top_category = self.categories[top5_catid[i]]
        #     top_prob = round(top5_prob[i].item(),4)
        #     top_categories[top_category] = top_prob

        return output


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


class CombinedModel(nn.Module):
    def __init__(self,
                 decoder: list = None):
        super(CombinedModel, self).__init__()
##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the combined model#
##############################################################
        
        self.decoder = decoder

    def forward(self, image_features, text_features):
        pass

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pass

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pass


# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# text_decoder.pkl                                           #
##############################################################
    pass
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    Image_model  = ImageClassifier()
    state_dict = torch.load("model_epoch79.pt")
    Image_model.load_state_dict(state_dict, strict= False)

    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the combined model#
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# combined_decoder.pkl                                       #
##############################################################
    pass
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the text processor that you will use to process #
# the text that you users will send to your API.             #
# Make sure that the max_length you use is the same you used #
# when you trained the model. If you used two different      #
# lengths for the Text and the Combined model, initialize two#
# text processors, one for each model                        #
##############################################################
    pass
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: TextItem):
  
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the text model   #
    # text.text is the text that the user sent to your API       #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
        "Category": "", # Return the category here
        "Probabilities": "" # Return a list or dict of probabilities here
            })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    predictions = Image_model(pil_image)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": f"{predictions}" # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)