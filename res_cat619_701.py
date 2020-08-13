import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.tanh=nn.Tanh()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        # residual = self.tanh(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class res_deahze(nn.Module):
    def __init__(self):
        super(res_deahze, self).__init__()
        scale_factor = 8
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=3, padding=1),
              nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)
        self.block10 = ResidualBlock(64)
        self.block11 = ResidualBlock(64)
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = ResidualBlock(64)
        self.block15 = ResidualBlock(64)
        self.block16 = ResidualBlock(64)
        self.block17 = ResidualBlock(64)
        self.block18 = ResidualBlock(64)
        self.block19 = ResidualBlock(64)
        self.block20 = ResidualBlock(64)
        self.block21 = ResidualBlock(64)
        self.block22 = ResidualBlock(64)
        # self.block23 = ResidualBlock(64)
        # self.block24 = ResidualBlock(64)
        # self.block25 = ResidualBlock(64)
        # self.block26 = ResidualBlock(64)
        # self.block27 = ResidualBlock(64)
        # self.block28 = ResidualBlock(64)
        # self.block29 = ResidualBlock(64)
        # self.block30 = ResidualBlock(64)
        # self.block31 = ResidualBlock(64)
        # self.block32 = ResidualBlock(64)
        # self.block33 = ResidualBlock(64)
        # self.block34 = ResidualBlock(64)
        self.block35 = ResidualBlock(64)
        self.block36 = ResidualBlock(64)
        self.block37 = ResidualBlock(64)
        self.block38 = ResidualBlock(64)
        self.block39 = ResidualBlock(64)
        self.block40 = ResidualBlock(64)
        self.block41 = ResidualBlock(64)
        self.block42 = ResidualBlock(64)
        self.block43 = ResidualBlock(64)
        self.block44 = ResidualBlock(64)
        self.block45 = ResidualBlock(64)
        self.block46 = ResidualBlock(64)
        self.block47 = ResidualBlock(64)
        self.block48 = ResidualBlock(64)
        self.block49 = ResidualBlock(64)
        self.block50 = ResidualBlock(64)
        self.block51 = ResidualBlock(64)
        self.block52 = ResidualBlock(64)
        self.block53 = ResidualBlock(64)
        self.block54 = ResidualBlock(64)
        self.block55 = ResidualBlock(64)
        self.block56 = ResidualBlock(64)
        self.block57 = ResidualBlock(64)
        self.block58 = ResidualBlock(64)
        self.block59 = ResidualBlock(64)
        self.block60 = ResidualBlock(64)
        self.block61 = ResidualBlock(64)
        # self.block62 = ResidualBlock(64)
        # self.block63 = ResidualBlock(64)
        # self.block64 = ResidualBlock(64)
        # self.block65 = ResidualBlock(64)
        # self.block66 = ResidualBlock(64)
        # self.block67 = ResidualBlock(64)
        # self.block68 = ResidualBlock(64)
        self.block69 = ResidualBlock(64)
        self.block70 = ResidualBlock(64)

        self.block71 = nn.Sequential(
              nn.Conv2d(128, 64, kernel_size=3, padding=1),
              nn.PReLU()
        )
        block72 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block72.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.block72 = nn.Sequential(*block72)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.pool(self.block2(block1))
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        block12 = self.block12(block11)
        block13 = self.block13(block12)
        block14 = self.block14(block13)
        block15 = self.block15(block14)
        block16 = self.block16(block15)
        block17 = self.block17(block16)
        block18 = self.block18(block17)
        block19 = self.block19(block18)
        block20 = self.block20(block19)
        block21 = self.block21(block20)
        block22 = self.block22(block21)
        # block23 = self.block23(block22)
        # block24 = self.block24(block23)
        # block25 = self.block25(block24)
        # block26 = self.block26(block25)
        # block27 = self.block27(block26)
        # block28 = self.block28(block27)
        # block29 = self.block29(block28)
        # block30 = self.block30(block29)
        # block31 = self.block46(block30)
        # block32 = self.block32(block31)
        # block33 = self.block33(block32)
        # block34 = self.block34(block33)
        block35 = self.upsample(self.block35(block22))
        block36 = self.pool(self.block36(block1))
        block37 = self.pool(self.block37(block36))
        block38 = self.block38(block37)
        block39 = self.block39(block38)
        block40 = self.block40(block39)
        block41 = self.block41(block40)
        block42 = self.block42(block41)
        block43 = self.block43(block42)
        block44 = self.block44(block43)
        block45 = self.block45(block44)
        block46 = self.block46(block45)
        block47 = self.block47(block46)
        block48 = self.block48(block47)
        block49 = self.block49(block48)
        block50 = self.block50(block49)
        block51 = self.block51(block50)
        block52 = self.block52(block51)
        block53 = self.block53(block52)
        block54 = self.block54(block53)
        block55 = self.block55(block54)
        block56 = self.block56(block55)
        block57 = self.block57(block56)
        block58 = self.block58(block57)
        block59 = self.block59(block58)
        block60 = self.block60(block59)
        block61 = self.block61(block60)
        # block62 = self.block62(block61)
        # block63 = self.block63(block62)
        # block64 = self.block64(block63)
        # block65 = self.block65(block64)
        # block66 = self.block66(block65)
        # block67 = self.block67(block66)
        # block68 = self.block68(block67)
        block69 = self.upsample(self.block69(block61))
        block70 = self.upsample(self.block70(block69))
        cat = torch.cat((block35, block70), 1)
        block71 = self.block71(cat)
        # block72 = self.block72(block1 + block71)
        block72 = self.block72(block71)

        return F.tanh(block72)