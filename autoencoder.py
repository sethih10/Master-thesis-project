import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    A convolutional autoencoder (AE) class for image reconstruction tasks.

    Parameters:
    -----------
    c : int
        The number of output channels for the first convolutional layer in the encoder.
    layer : int, optional
        The number of encoding/decoding layers. Default is 2, and it can be set to 3 for an additional layer.

    Methods:
    --------
    forward(x)
        Forward pass through the autoencoder.

    """

    def __init__(self, c, layer=2):
        super(Autoencoder, self).__init__()
        
        encoder_layers = [
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0),
            nn.Conv2d(c, c*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        ]

        if layer == 3:
            encoder_layers.extend([
                nn.Conv2d(c*2, c*4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), padding=0)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        
        if layer == 3:
            decoder_layers.extend([
                nn.ConvTranspose2d(c*4, c*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ])
        
        decoder_layers.extend([
            nn.ConvTranspose2d(c*2, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
