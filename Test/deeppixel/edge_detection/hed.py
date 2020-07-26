import cv2
import os

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self,inputs):
        '''the crop layer will receive two inputs -- we need to crop the first 
        input blob to match the shape of the second one,keeping the batch size 
        and number of channels.Then we compute the starting and ending crop coordinates.
        We get the shape of the volume and use the derived coordinates to perform the crop
        in the forward function
        '''
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

def image_to_HED(path):
    '''
    Here we load all the pretrained models and register the CropLayer class in it.
    Then we construct a blob from the image and pass it to the net.Then we resize the 
    output to our desired shape and 0-255 colour scale and ensure that its of uint8 type.
    '''
    protoPath = 'deeppixel\edge_detection\hed_model\deploy.prototxt'
    modelPath = 'deeppixel\edge_detection\hed_model\hed_pretrained_bsds.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    image = cv2.imread(path)
    cv2.dnn_registerLayer("Crop", CropLayer)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
    net.setInput(blob)
    hed_img = net.forward()
    hed_img = cv2.resize(hed_img[0, 0], (W, H))
    hed_img = (255 * hed_img).astype("uint8")
    return hed_img

def folder_to_HED(path):
    final=[]
    for file in os.listdir(path):
        img=image_to_HED(file)
        final.append(img)               
    return final    


