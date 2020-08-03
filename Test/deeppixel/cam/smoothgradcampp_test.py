import torchvision.models as models

from deeppixel.cam import *


image_path = 'Test/deeppixel/cam/bull_mastiff.png'
#image_name = image_path.split('/')[1]
image = load_image(image_path)
norm_image = apply_transforms(image)

model = models.vgg16(pretrained=True)
model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))

smoothgradcampp = SmoothGradCAMpp(model_dict)
output = smoothgradcampp(norm_image)
visualize(norm_image, output)