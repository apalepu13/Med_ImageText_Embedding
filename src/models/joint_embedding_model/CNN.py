import torch
from torch import nn
import time

class CNN_Embeddings(nn.Module):
    def __init__(self, embed_dim, freeze = False):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, embed_dim)
        #summary(self.resnet, (3, 224, 224))
    def forward(self, image):
        #t = time.time()
        im_embeddings = self.resnet(image)
        #print("CNN batch time " + str(time.time() - t))
        return im_embeddings

#cn = CNN_Embeddings(100)



