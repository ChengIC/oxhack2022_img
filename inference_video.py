import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
class_names=['can', 'chips_bag', 'Unknown']

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=[512,512]),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
MODEL='model.pth'
model = torch.load(MODEL,map_location ='cpu')
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    original_frame_copy=frame.copy()
    with torch.no_grad():
        img=Image.fromarray(np.uint8(frame)).convert('RGB')
        inputs=preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) 
        label=class_names[preds]
    image = cv2.putText(original_frame_copy, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('preview',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print ('pred result is: '+ str(label))
    