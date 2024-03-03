import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        in_channel = 3
        channel_width_list = [8, 16, 32, 64]

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, channel_width_list[0], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(channel_width_list[0], channel_width_list[1], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(channel_width_list[1], channel_width_list[2], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(channel_width_list[2], channel_width_list[3], kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        out_channel = 3
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channel_width_list[3], channel_width_list[2], kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_width_list[2], channel_width_list[2], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(channel_width_list[2], channel_width_list[1], kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_width_list[1], channel_width_list[1], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(channel_width_list[1], channel_width_list[0], kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_width_list[0], out_channel, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        feat = self.encoder(x)
        ret = self.decoder(feat)

        return ret

if __name__ == '__main__':
    print('tiny model test')