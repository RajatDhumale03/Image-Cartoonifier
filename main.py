import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

#Loading the model vgg19 that will serve as the base model
model=models.vgg19(pretrained=True).features

class VGG(nn.Module):
    def _init_(self):
        super(VGG, self).__init__()
        self.chosen_features= ['0','5','10','19','28']
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers


    #x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        #initialize an array that wil hold the activations from the chosen layers
        features=[]
        #Iterate over all the layers of the mode
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.chosen_features):
                features.append(x)

        return features

#defing a function that will load the image and perform the required preprocessing and put it on the GPU
def load_image(image_name):
    image=Image.open(image_name)
    #defining the image transformation steps to be performed before feeding them to the model
    image=loader(image).unsqueeze(0)
    return image.to(device)

#Assigning the GPU to the variable device
device=torch.device( "cuda" if (torch.cuda.is_available()) else "cpu")
image_size= 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalise(mean=[],std=[])
    ]
)

#Loading the original and the style image
original_img=load_image('raj1.png')
style_img=load_image('style1.png')

model = VGG.to(device).eval()
#Creating the generated image from the original image
generated=original_img.clone().requires_grad_(True)

#Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        #Compute Gram Matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A)**2)

    total_loss = alpha*original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")
