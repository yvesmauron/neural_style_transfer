import torch
import torch.nn as nn
from tl.losses import ContentLoss, StyleLoss
from tl.preprocess import Normalization

import logging
import copy
# for this quick and dirty project; this is enough :-)
logging.basicConfig(
    level=logging.DEBUG,
    format="'%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


class NeuralTransferTrainer(object):
    """Trains a neural transfer leanring moder"""
    def __init__(
        self,
        model,
        normalization_mean,
        normalization_std,
        optimizer,
        content_image,
        input_image,
        style_image,
        content_layers=['conv_4'],
        style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
        device="cuda"
        ):
        """Cosstructor
        
        Arguments:
            model {troch.nn} -- Cnn model to be used
            normalization_mean {troch.tensor} -- Mean to be used to standardize input image
            normalization_std {torch.tensor} -- Std to be used to standardize input image
            optimizer {torch.optim} -- Optimizer to be used
            content_image {troch.tensor} -- Content image
            style_image {torch.tensor} -- Style image to be used
            input_image {torch.tensor} -- Input image to be used - usually the same as content image
        
        Keyword Arguments:
            content_layers {list} -- Layers to be used for content sloss (default: {['conv_4']})
            style_layers {list} -- Layers to be used for style loss (default: {['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']})
            device {str} -- Which device should be used (default: {"cuda"})
        
        Raises:
            RuntimeError: [description]
        
        Returns:
            [type] -- [description]
        """
        super(NeuralTransferTrainer, self).__init__()
        self.cnn = copy.deepcopy(model)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.optimizer = optimizer
        self.content_image = content_image
        self.style_image = style_image
        self.input_image = input_image
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.device = device


        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        # losses
        self.content_losses = []
        self.style_losses = []
        # assuming that self.cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        self.model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(self.content_image).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.model(self.style_image).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]
    
    def restyle(
        self,
        n_epochs=300,
        style_weight=1000000, 
        content_weight=1
        ):

        run = [0]
        while run[0] <= n_epochs:

            def closure():
                # correct the values of updated input image
                self.input_image.data.clamp_(0, 1)

                self.optimizer.zero_grad()
                self.model(self.input_image)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    log_str = "Epoch {:3d}      Style Loss: {:08.4f}        Content Loss: {:07.4f}".format(
                        run[0],
                        style_score.item(),
                        content_score.item()
                    )
                    logging.info(log_str)

                return style_score + content_score

            self.optimizer.step(closure)

        # a last correction...
        self.input_image.data.clamp_(0, 1)

        return self.input_image