import torch
# working with images and plotting
from PIL import Image
import torchvision.transforms as transforms


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def image_loader(image_name, loader, device):
    image = Image.open(image_name)
    # add batch dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)