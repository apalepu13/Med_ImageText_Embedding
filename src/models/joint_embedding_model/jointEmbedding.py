from torch import nn
from CNN import *
from Transformer import *
import numpy as np
import time

class JointEmbeddingModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = CNN_Embeddings(embed_dim)
        self.transformer = Transformer_Embeddings(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))

    def forward(self, image, text):
        #t = time.time()
        image_features = self.cnn(image)
        text_features = self.transformer(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        #print("cosine similarity time: " + str(time.time() - t))

        return logits_per_image, logits_per_text