# Step 1 : Import the required libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import argparse


# Step 2: Load the VGG network
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

# if GPU available, move the model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
#print(vgg)


#Step 3: Load the images
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image


# Step 4: Download Required Content and Style Images
args = argparse.ArgumentParser()
args.add_argument("--path_to_content_image", help="link_to_content_image", default='./input/usa.jpg')
args.add_argument("--path_to_style_image", help="link_to_style_image", default='./input/starrynight.jpg')
args.add_argument("--path_to_save_output_file", help = "path_to_save_output_file", default='./output')
args = args.parse_args()

content = load_image(args.path_to_content_image).to(device)
style = load_image(args.path_to_style_image, shape=content.shape[-2:]).to(device)

# Step 5 : View the downloaded images
def im_convert(tensor):    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.set_title("Content Image",fontsize = 20)
ax2.imshow(im_convert(style))
ax2.set_title("Style Image", fontsize = 20)
plt.show()


# Step 6: Extract features from the image
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


# Step 7: Define a gram martrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())   
    return gram 



# Step 8 : Extract the content and style features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
target = content.clone().requires_grad_(True).to(device)



# Step 9: Define the loss
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e9  # beta


show_every = 400
optimizer = optim.Adam([target], lr=0.003)
steps = 400

for ii in range(1, steps+1):
    
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0

    for layer in style_weights:
   
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)
        

    total_loss = content_weight * content_loss + style_weight * style_loss
    
    print(ii)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()




# Step 10: Display the output

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
ax1.imshow(im_convert(content))
ax1.set_title("Content Image", fontsize = 20)
ax2.imshow(im_convert(target))
ax2.set_title("Stylized Target Image", fontsize = 20)
ax1.grid(False)
ax2.grid(False)

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()


#Step 11: Save output image
output_img = im_convert(target)
output_img = output_img*255
output_img = Image.fromarray(output_img.astype('uint8'))
output_img.save("{}/Stylized_Target_Image.jpg".format(args.path_to_save_output_file))