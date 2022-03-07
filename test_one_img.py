import torch
from PIL import Image
from torchvision import transforms

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

img_file = 'val_imgs/chips_bag/WechatIMG53.jpeg'
with torch.no_grad():
    img=Image.open(img_file).convert('RGB')
    inputs=preprocess(img).unsqueeze(0).to(device)
    outputs = model(inputs)
    print (outputs)
    _, preds = torch.max(outputs, 1) 
    label=class_names[preds]
    print ('pred result is: '+ str(label))