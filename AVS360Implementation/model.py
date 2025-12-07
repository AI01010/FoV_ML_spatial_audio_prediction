import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.resnet3D import resnet18
from utils.resnet3D_cubic import resnet18 as resnet18_cubic
#from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi
import pdb


class ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(ScaleUp, self).__init__()

        self.combine = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)

        #self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.combine.weight)
        nn.init.constant_(self.combine.bias, 0.0)

    def forward(self, inputs):
        output = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.combine(output)
        output = F.relu(output, inplace=True)
        return output


class DAVE(nn.Module):

    def __init__(self):
        super(DAVE, self).__init__()

        self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=16, num_classes=12, last_fc=False, last_pool=True)
        self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False, last_pool=False)
        self.video_branch_cubic = resnet18_cubic(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False, last_pool=False)

        self.upscale1 = ScaleUp(512, 512)
        self.upscale2 = ScaleUp(512, 128)
        self.combinedEmbedding = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.combinedEmbedding_equi_cp = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        #self._weights_init()
        self.saliency = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        
        self.c2e = Cube2Equi(4)   # input h of equi_img
        self.w = 0.5
        
    def _weights_init(self):   
        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)
        nn.init.kaiming_normal_(self.combinedEmbedding.weight)
        nn.init.constant_(self.combinedEmbedding.bias, 0.0)
        
    def forward(self, v_equi, v_cube, a, aem, eq_b):
        B, num_faces, C, T, Hc, Wc = v_cube.shape  # v_cube = [B,6,3,16,128,128]

        # --- Equi branch ---
        xV1_equi = self.video_branch(v_equi)  # [B, 512, 1, 8, 16]

        # --- Cube branch ---
        xV1_cube = v_cube.view(B*num_faces, C, T, Hc, Wc)  # Merge batch & faces
        xV1_cube = self.video_branch_cubic(xV1_cube)       # [B*num_faces, 512, 1, H', W']
        xV1_cube = xV1_cube.view(B, num_faces, 512, xV1_cube.size(2), xV1_cube.size(3), xV1_cube.size(4))
        xV1_cube = torch.squeeze(xV1_cube, 3)             # [B, num_faces, 512, H', W']

        # Convert cube faces to equirectangular
        xV1_cube_equi = self.c2e.to_equi_nn(xV1_cube)     # [B, 512, Hc', Wc']

        # --- Fix interpolation ---
        xV1_cube_equi_2d = xV1_cube_equi                  # shape [B, 512, Hc', Wc']
        xV1_cube_equi_2d = F.interpolate(
            xV1_cube_equi_2d,
            size=(xV1_equi.size(3), xV1_equi.size(4)),   # target H, W from equi branch
            mode='bilinear',
            align_corners=True
        )
        xV1_cube_equi = xV1_cube_equi_2d.unsqueeze(2)    # add back temporal dimension: [B, 512, 1, H, W]

        # Combine video branches
        xV1_combined = xV1_equi*self.w + xV1_cube_equi*(1-self.w)

        # --- Audio branch ---
        xA1 = self.audio_branch(a)                        # [B,512,1,1,1]
        xA1 = xA1.expand_as(xV1_combined)                # Broadcast to [B,512,1,H,W]

        # Combine audio and video and reduce to 4D
        xC = torch.cat((xV1_equi, xA1), dim=1)  # [B,1024,1,H,W]
        xC = torch.squeeze(xC, dim=2)           # [B,1024,H,W]
        x = self.combinedEmbedding(xC)          # [B,512,H,W]

        # Broadcast eq_b and aem
        if eq_b.dim() == 3: eq_b = eq_b.unsqueeze(1)
        if aem.dim() == 3: aem = aem.unsqueeze(1)

        x = eq_b * (x + x.max()) * (1.0 + aem)  # [B,512,H,W]
        x = F.relu(x, inplace=True)

        x = x.squeeze(1)  # remove the extra dimension, now x = [B, C, H, W]
        # print("DEBUG: x shape after combinedEmbedding and eq_b/aem:", x.shape)
        # print("DEBUG: x.dim() =", x.dim())

        # Only now pass to ScaleUp
        x = self.upscale1(x)  # [B,512,H*2,W*2]
        x = self.upscale2(x)  # [B,128,H*4,W*4]

        sal = self.saliency(x)                                         # x = [10, 1, 32, 64]
        sal = F.relu(sal, inplace=True)                                # x = [10, 1, 32, 64]
        
        sal = sal/sal.view(sal.size(0),-1).sum(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        return sal