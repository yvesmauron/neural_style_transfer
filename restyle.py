from __future__ import print_function

# torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# working with images and plotting
from PIL import Image

# torch vision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# import trainer
from tl.trainer import NeuralTransferTrainer
from tl.utils import image_loader

# logging
import logging
from datetime import datetime

# args
import argparse

# file handling
import os
from pathlib import Path

# -------------------
# logging
# -------------------
# for this quick and dirty project; this is enough :-)
logging.basicConfig(
    level=logging.DEBUG,
    format="'%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# -------------------
# Parsing arguments
# -------------------
parser = argparse.ArgumentParser(description="Neural transfer learnig")

parser.add_argument(
    "--content_image",
    default="./data/content/me.jpg",
    help="Path to your content files"
)

parser.add_argument(
    "--style_image",
    default="./data/style/style.JPG",
    help="Path to your style files"
)

parser.add_argument(
    "--restyled_path",
    default="./data/restyled/",
    help="Path to your target directory"
)

parser.add_argument(
    "--restyled_image_name",
    default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    help="Name of the restyled image"
)

# creating folder paths for content, style and restyled images
args = parser.parse_args()


if __name__ == "__main__":

    # -------------------
    # get inputs
    # -------------------
    content_image = str(args.content_image)
    style_image = str(args.style_image)
    restyled_path = str(args.restyled_path)
    restyled_image_path = os.path.join(restyled_path, str(args.restyled_image_name) + ".png")

    # -------------------
    # Checking input
    # -------------------
    if not os.path.exists(content_image):
        logging.error("Proper folder structure not in place, please setup the folder structure using setup.py")
        raise FileNotFoundError

    if not os.path.exists(style_image):
        logging.error("Proper folder structure not in place, please setup the folder structure using setup.py")
        raise FileNotFoundError

    if not os.path.exists(restyled_path):
        logging.info("Creating folder for restyled images: {}".format(restyled_path))
        Path(restyled_path).mkdir(parents=True, exist_ok=True)
    else:
        logging.info("Restyle folder {} exists.".format(restyled_path))
    
    if os.path.exists(restyled_image_path):
        logging.error("There is already a file with path: {}".format(restyled_image_path))
        raise FileExistsError


    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------
    # Loading the Images
    # -------------------

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    # define transform loader
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    # load images
    style_img = image_loader(style_image, loader, device)
    content_img = image_loader(content_image, loader, device)

    # check if style image and input image is the same
    if style_img.size() != content_img.size():
        logging.error("Content and style image need to be the same size")
        raise ValueError

    # clone content image for input
    input_img = content_img.clone()

    # -------------------
    # Starter model
    # -------------------
    # starter model and required normalization
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # -------------------
    # Optimizer
    # -------------------
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    # -------------------
    # Define trainer
    # -------------------
    model_trainer = NeuralTransferTrainer(
        model=cnn,
        normalization_mean=cnn_normalization_mean,
        normalization_std=cnn_normalization_std,
        optimizer=optimizer,
        content_image=content_img,
        input_image=input_img,
        style_image=style_img,
        content_layers=['conv_4'], # layer to comput content loss
        style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], # layer to compute style loss
        device="cuda"
    )

    # -------------------
    # Train model (restyle image)
    # -------------------
    output = model_trainer.restyle(
        n_epochs =300,
        style_weight=1000000, 
        content_weight=1
    )

    # -------------------
    # Save restyled image
    # -------------------
    save_image(output.squeeze(), fp= restyled_image_path)
