#import base64
import requests

url = 'http://0.0.0.0:8080/predict/image'
# with open("formated_images/-8cdf-438f-9b4c-1672631747fd.jpg", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read())
    
# payload ={"filename": "API_model_image.jpg", "filedata": encoded_string}

file = {'image': open('formated_images/ff933077-7144-49a6-9c7e-42950cc3d319.jpg', 'rb')}
resp = requests.post(url=url, files=file)
print(resp.text)