import numpy as np
import cv2
import math as m
import torch
from torch import nn
from torch.autograd import Variable
from utils.sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from utils.sph_utils import face_to_cube_coord, norm_to_cube
import pdb

class Cube2Equi:
    def __init__(self, input_w):
        scale_c = 1
        in_width = input_w * scale_c
        out_w = in_width * 4
        out_h = in_width * 2
        out_arr = np.zeros((out_h, out_w, 3), dtype='float32')

        face_map = np.zeros((out_h, out_w))  # for face indexing

        XX, YY = np.meshgrid(range(out_w), range(out_h))  # for output grid

        theta, phi = xy2angle(XX, YY, out_w, out_h)
        theta = pruned_inf(theta)
        phi = pruned_inf(phi)

        _x, _y, _z = to_3dsphere(theta, phi, 1)
        face_map = get_face(_x, _y, _z, face_map)
        x_o, y_o = face_to_cube_coord(face_map, _x, _y, _z)

        out_coord = np.transpose(np.array([x_o, y_o]), (1, 2, 0))  # h x w x 2
        out_coord = norm_to_cube(out_coord, in_width)

        self.out_coord = out_coord
        self.face_map = face_map

    def to_equi_nn(self, input_data):
        ''' 
        input_data: [batch, 6, depth, w, w]
        out: [batch, depth, out_h, out_w]
        '''
        # Prepare numpy --> torch tensors on correct device
        device = input_data.device
        gridf = torch.tensor(self.out_coord, dtype=torch.float32, device=device).contiguous()   # [out_h, out_w, 2]
        face_map = torch.tensor(self.face_map.astype(np.int64), dtype=torch.long, device=device).contiguous()  # [out_h, out_w]

        out_h, out_w = gridf.shape[0], gridf.shape[1]
        batch_size = input_data.size(0)
        depth = input_data.size(2)

        # Normalize gridf to [-1, 1] (same scale for both dims)
        max_val = torch.max(gridf)
        gridf = (gridf - (max_val / 2.0)) / (max_val / 2.0)   # preserves original intent

        # Make grid batch-sized: [batch, out_h, out_w, 2]
        gridf = gridf.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Prepare output tensor (no in-place ops; create on same device)
        warp_out = torch.zeros((batch_size, depth, out_h, out_w), dtype=torch.float32, device=device)

        # loop over cube faces and fill warp_out where mask==face
        for f_idx in range(6):
            # boolean mask [out_h, out_w]
            face_mask = (face_map == f_idx)            # dtype=torch.bool
            if not face_mask.any():
                continue

            # mask shape -> [batch, depth, out_h, out_w] via broadcasting
            mask = face_mask.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
            mask = mask.expand(batch_size, depth, out_h, out_w) # [N,C,H,W]

            # sampled_face: sample from face f_idx
            # input_data[:, f_idx] has shape [batch, depth, H_in, W_in]
            # gridf has shape [batch, out_h, out_w, 2]
            sampled_face = nn.functional.grid_sample(
                input_data[:, f_idx],    # [N, C, H_in, W_in]
                gridf,                   # [N, H_out, W_out, 2]
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # returned shape: [N, C, H_out, W_out]

            # place sampled_face into warp_out, out-of-place
            warp_out = torch.where(mask, sampled_face, warp_out)

        return warp_out

    def to_equi_cv2(self, input_data):
        ''' 
        input_data: 6 * w * w * c
        gridf: 2w * 4w * 2
        face_map: 2w * 4w
        output: 1 * 2w * 4w * c
        '''
        gridf = self.out_coord
        face_map = self.face_map
        out_w = gridf.shape[1]
        out_h = gridf.shape[0]

        in_width = out_w/4
        depth = input_data.shape[1]

        gridf = gridf.astype(np.float32)
        out_arr = np.zeros((out_h, out_w, depth), dtype='float32')
        input_data = np.transpose(input_data, (0, 2, 3, 1))

        for f_idx in range(0, 6):
            for dept in range(int(1000/4)):
                out_arr[face_map == f_idx, 4*dept:4*(dept+1)] = cv2.remap(input_data[f_idx, :, :, 4*dept:4*(
                    dept+1)], gridf[:, :, 0], gridf[:, :, 1], cv2.INTER_CUBIC)[face_map == f_idx]
        return np.transpose(out_arr, (2, 0, 1))
