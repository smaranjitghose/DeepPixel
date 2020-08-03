import torchvision.models as models

from deeppixel.cam import *


image_path = 'Test/deeppixel/cam/bull_mastiff.png'
#image_name = image_path.split('/')[1]
image = load_image(image_path)
norm_image = apply_transforms(image)

model = models.vgg16(pretrained=True)
model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))

sscam1 = SSCAM1(model_dict)
output = sscam1(norm_image)
visualize(norm_image, output)

sscam2 = SSCAM2(model_dict)
output = sscam2(norm_image)
visualize(norm_image, output)