# Z_RISINGConvNet_3D.py

from typing import Optional
import torch
import torch.nn as nn
from LION.models import LIONmodel
from LION.models.LIONmodel import LIONModelParameter
import LION.CTtools.ct_geometry as ct


class ConvBlock(nn.Module):
    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3, norm_type="group"):
        super().__init__()
        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")

        layer_list = []
        pad = kernel_size // 2

        for ii in range(layers):
            in_ch = channels[ii]
            out_ch = channels[ii + 1]

            layer_list.append(
                nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
            )

            if norm_type == "group":
                num_groups = min(8, out_ch)
                while out_ch % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                layer_list.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
            elif norm_type == "batch":
                layer_list.append(nn.BatchNorm3d(out_ch))
            else:
                raise ValueError(f"Unknown norm_type {norm_type}")

            if ii < layers - 1 or relu_last:
                if relu_type == "ReLU":
                    layer_list.append(nn.ReLU(inplace=True))
                elif relu_type == "LeakyReLU":
                    layer_list.append(nn.LeakyReLU(inplace=True))
                elif relu_type != "None":
                    raise ValueError("Wrong ReLU type " + relu_type)

        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        return self.pool(x)




### Upsample + Conv3d model here ###
class Up(nn.Module):
    def __init__(self, channels, stride=2, relu_type="ReLU", norm_type="group"):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        conv = nn.Conv3d(
            channels[0],
            channels[1],
            kernel_size=3,
            padding=1,
            bias=False
        )

        layers = [self.up, conv]

        out_ch = channels[1]

        if norm_type == "group":
            num_groups = min(8, out_ch)
            while out_ch % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm3d(out_ch))

        if relu_type == "ReLU":
            layers.append(nn.ReLU(inplace=True))
        elif relu_type == "LeakyReLU":
            layers.append(nn.LeakyReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class RISINGConvNet_3D(LIONmodel.LIONmodel):
    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter] = None,
        geometry: Optional[ct.Geometry] = None,
    ):
        assert geometry is not None, "Geometry parameters required for RISINGConvNet_3D."
        super().__init__(model_parameters, geometry)
        self._make_operator()

        self.block_1_down = ConvBlock(
            self.model_parameters.down_1_channels,
            relu_type=self.model_parameters.activation
        )
        self.down_1 = Down()

        self.block_2_down = ConvBlock(
            self.model_parameters.down_2_channels,
            relu_type=self.model_parameters.activation
        )
        self.down_2 = Down()

        self.block_3_down = ConvBlock(
            self.model_parameters.down_3_channels,
            relu_type=self.model_parameters.activation
        )
        self.down_3 = Down()

        self.block_4_down = ConvBlock(
            self.model_parameters.down_4_channels,
            relu_type=self.model_parameters.activation
        )
        self.down_4 = Down()

        self.block_bottom = ConvBlock(
            self.model_parameters.latent_channels,
            relu_type=self.model_parameters.activation
        )

        self.up_1 = Up(
            [self.model_parameters.latent_channels[-1], self.model_parameters.up_1_channels[0] // 2],
            relu_type=self.model_parameters.activation
        )
        self.block_1_up = ConvBlock(
            self.model_parameters.up_1_channels,
            relu_type=self.model_parameters.activation
        )

        self.up_2 = Up(
            [self.model_parameters.up_1_channels[-1], self.model_parameters.up_2_channels[0] // 2],
            relu_type=self.model_parameters.activation
        )
        self.block_2_up = ConvBlock(
            self.model_parameters.up_2_channels,
            relu_type=self.model_parameters.activation
        )

        self.up_3 = Up(
            [self.model_parameters.up_2_channels[-1], self.model_parameters.up_3_channels[0] // 2],
            relu_type=self.model_parameters.activation
        )
        self.block_3_up = ConvBlock(
            self.model_parameters.up_3_channels,
            relu_type=self.model_parameters.activation
        )

        self.up_4 = Up(
            [self.model_parameters.up_3_channels[-1], self.model_parameters.up_4_channels[0] // 2],
            relu_type=self.model_parameters.activation
        )
        self.block_4_up = ConvBlock(
            self.model_parameters.up_4_channels,
            relu_type=self.model_parameters.activation
        )

        self.block_last = nn.Conv3d(
            self.model_parameters.last_block[0],
            self.model_parameters.last_block[1],
            self.model_parameters.last_block[2],
            padding=0
        )

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()

        params.down_1_channels = [1, 8, 8, 8]
        params.down_2_channels = [8, 16, 16]
        params.down_3_channels = [16, 32, 32]
        params.down_4_channels = [32, 64, 64]

        params.latent_channels = [64, 128, 128]

        params.up_1_channels = [128, 64, 64]
        params.up_2_channels = [64, 32, 32]
        params.up_3_channels = [32, 16, 16]
        params.up_4_channels = [16, 8, 8]

        params.last_block = [8, 1, 1]
        params.activation = "ReLU"
        params.model_input_type = LIONmodel.ModelInputType.IMAGE

        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("RISINGConvNet_3D: lightweight 3D residual U-Net style post-processing model.")
        elif cite_format == "bib":
            print("@misc{risingconvnet3d, title={RISINGConvNet_3D}, note={Lightweight 3D residual U-Net style model}}")
        else:
            print("cite not implemented for selected method")

    def forward(self, x):
        image = x

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)

        return image + res