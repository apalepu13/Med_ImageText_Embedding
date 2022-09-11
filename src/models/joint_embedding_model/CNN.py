import torch
from torchsummary import summary
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_Embeddings(nn.Module):
    '''
    Resnet-50 architecture that outputs a flattened list of patch embedding of size (N, E, P)
    '''
    def __init__(self, embed_dim, freeze = False, imagenet = True, pool=False, args=None):
        super().__init__()
        self.imagenet = imagenet
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = not freeze

        laylist = list(self.resnet.children())[:8]
        self.resnet = nn.Sequential(*laylist)
        self.resnet.fc = nn.Conv2d(2048, embed_dim, kernel_size=1, bias=True)
        self.avgpooler = nn.AdaptiveAvgPool2d(1)
        self.pool = pool
        if not imagenet:
            state_dict = torch.load('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/biovil_image_resnet50_proj_size_128.pt', map_location = device)
            self.resnet.load_state_dict(state_dict)

    def forward(self, image):
        if isinstance(image, list):
            len_ims = len(image)
            if len_ims == 1:
                image = image[0].to(device)
            else:
                image = torch.cat(image, dim=0).to(device)
        im_embeddings = self.resnet(image)
        if self.imagenet:
            if not self.pool:
                patch_embeddings = im_embeddings.view(im_embeddings.shape[0], im_embeddings.shape[1],
                                                      im_embeddings.shape[2] * im_embeddings.shape[3])
                return patch_embeddings #N E P
            else:
                global_embeddings = self.avgpooler(im_embeddings)
                return global_embeddings
        else:
            return im_embeddings


class CNN_Classifier(nn.Module):
    '''
    Takes in a given CNN embedding model, or creates a new one
    Computes avg patch embedding and adds a classification head
    '''
    def __init__(self, embed_size,cnn_model=None, freeze=True, num_heads=5, args=None):
        super().__init__()
        self.cnn_model = cnn_model if cnn_model else CNN_Embeddings(embed_size)
        for param in self.cnn_model.parameters():
            param.requires_grad = not freeze and cnn_model
        self.relu = nn.ReLU()
        self.classification_head = nn.Linear(embed_size, num_heads)

    def forward(self, image):
        embeddings = self.cnn_model.resnet(image)
        embedding = self.cnn_model.avgpooler(embeddings)
        embedding = torch.squeeze(embedding)
        output = self.relu(embedding)
        output = self.classification_head(output)
        return output

if __name__=='__main__':
    cnn = CNN_Embeddings(128, imagenet=False)
    print(summary(cnn, (3, 224, 224)))

