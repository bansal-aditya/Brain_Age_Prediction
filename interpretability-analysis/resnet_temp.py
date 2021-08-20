import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Conv3d:
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # ResBlock 0
        self.res_block_0 = self.res_block(1, 8)
        self.skip_0 = self.skip_connection(1, 8)
        self.combine_0 = self.ELU_and_Pool()

        # ResBlock 1
        self.res_block_1 = self.res_block(8, 16)
        self.skip_1 = self.skip_connection(8, 16)
        self.combine_1 = self.ELU_and_Pool()

        # ResBlock 2
        self.res_block_2 = self.res_block(16, 32)
        self.skip_2 = self.skip_connection(16, 32)
        self.combine_2 = self.ELU_and_Pool()

        # ResBlock 3
        self.res_block_3 = self.res_block(32, 64)
        self.skip_3 = self.skip_connection(32, 64)
        self.combine_3 = self.second_last_ELU_and_Pool()

        # ResBlock 4
        self.res_block_4 = self.res_block(64, 128)
        self.skip_4 = self.skip_connection(64, 128)
        self.combine_4 = self.final_ELU_and_Pool()

        self.classifier = nn.Sequential(nn.Flatten())
        self.classifier.add_module("Final Layer", self.final_layer())

        self.apply(init_weights)

    def res_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
        )
        return block

    def skip_connection(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0))

    def ELU_and_Pool(self):
        return nn.Sequential(nn.ReLU(),
                             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1))

    def second_last_ELU_and_Pool(self):
        return nn.Sequential(nn.ReLU(),
                             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

    def final_ELU_and_Pool(self):
        return nn.Sequential(nn.ReLU(),
                             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0))

    def final_layer(self):
        return nn.Sequential(
            nn.Linear(10240, 128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Linear(128, 1))

    def forward(self, x):

        # ResBlock 0
        inp = x
        x = self.res_block_0(x)
        skip = self.skip_0(inp)
        x = x + skip
        x = self.combine_0(x)

        # ResBlock 1
        inp = x
        x = self.res_block_1(x)
        skip = self.skip_1(inp)
        x = x + skip
        x = self.combine_1(x)

        # ResBlock 2
        inp = x
        x = self.res_block_2(x)
        skip = self.skip_2(inp)
        x = x + skip
        x = self.combine_2(x)

        # ResBlock 3
        inp = x
        x = self.res_block_3(x)
        skip = self.skip_3(inp)
        x = x + skip
        x = self.combine_3(x)

        # ResBlock 4
        inp = x
        x = self.res_block_4(x)
        skip = self.skip_4(inp)
        x = x + skip
        x = self.combine_4(x)

        # Classifier
        x = self.classifier(x)

        return x

