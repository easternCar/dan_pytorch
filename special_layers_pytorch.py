import torch
from torch.nn.modules.utils import _pair

class AffineImageTransformation(torch.nn.Module):
    def __init__(self, img_size):

        super().__init__()
        self.img_size = _pair(img_size)

    def forward(self, input_image, affine_params):

        affine_params = affine_params.view(-1, 2, 3)
        affine_grid = torch.nn.functional.affine_grid(
            affine_params, (input_image.size(
                0), input_image.size(1), *self.img_size))

        return torch.nn.functional.grid_sample(input_image, affine_grid)


class AffineLandmarkTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lmk_tensor, affine_tensor, inverse=False):

        A = torch.zeros((affine_tensor.size(0), 2, 2),
                        device=affine_tensor.device)

        A[:, 0] = affine_tensor[:, :2].clone()
        A[:, 1] = affine_tensor[:, 2:4].clone()
        t = affine_tensor[:, 4:6].clone()

        if inverse:
            A = A.inverse()
            t = torch.bmm(
                (-t).view(affine_tensor.size(0), -1, 2), A.permute(0, 2, 1))

            t = t.squeeze(1)

        output = torch.bmm(lmk_tensor.view(affine_tensor.size(0), -1, 2), A)

        t = t.unsqueeze(1)

        output = output + t

        output = output.view(affine_tensor.size(0), -1, 2)

        return output


import torch


class EstimateAffineParams(torch.nn.Module):
    def __init__(self, mean_shape):

        super().__init__()
        self.register_buffer("mean_shape", mean_shape)
        #self.mean_shape = mean_shape            # no register?

    def forward(self, transformed_shape):

        source = transformed_shape.view((transformed_shape.size(0), -1, 2))
        batch_size = source.size(0)

        dst_mean = self.mean_shape.mean(dim=0)
        src_mean = source.mean(dim=1)

        dst_mean = dst_mean.unsqueeze(dim=0)
        src_mean = src_mean.unsqueeze(dim=1)

        src_vec = (source - src_mean).view(batch_size, -1)
        dest_vec = (self.mean_shape - dst_mean).view(-1)
        dest_vec = dest_vec.expand(batch_size, *dest_vec.shape)

        dest_norm = torch.zeros(batch_size, device=dest_vec.device)
        src_norm = torch.zeros(batch_size, device=src_vec.device)

        for i in range(batch_size):
            dest_norm[i] = dest_vec[i].norm(p=2)
            src_norm[i] = src_vec[i].norm(p=2)

        a = torch.bmm(src_vec.view(batch_size, 1, -1),
                      dest_vec.view(batch_size, -1, 1)).squeeze()/src_norm**2
        b = 0

        for i in range(self.mean_shape.shape[0]):
            b += src_vec[:, 2*i] * dest_vec[:, 2*i+1] - \
                src_vec[:, 2*i+1] * dest_vec[:, 2*i]
        b = b / src_norm**2

        A = torch.zeros((batch_size, 2, 2), device=a.device)
        A[:, 0, 0] = a
        A[:, 0, 1] = b
        A[:, 1, 0] = -b
        A[:, 1, 1] = a

        src_mean = torch.bmm(src_mean.view(batch_size, 1, -1), A)
        out = torch.cat(
            (A.view(batch_size, -1), (dst_mean - src_mean).view(batch_size, -1)), 1)

        return out

import torch
import itertools


class HeatMap(torch.nn.Module):

    def __init__(self, img_size, patch_size):


        super().__init__()

        self.img_shape = _pair(img_size)
        self.half_size = patch_size // 2

        offsets = torch.tensor(
            list(
                itertools.product(
                    range(-self.half_size, self.half_size + 1),
                    range(-self.half_size, self.half_size + 1)
                )
            )
        ).float()

        self.register_buffer("offsets",
                             offsets
                             )

    # landmark : x_aff_lms
    def draw_lmk_helper(self, landmark):

        img = torch.zeros(1, *self.img_shape, device=landmark.device)

        int_lmk = landmark.to(torch.long)
        locations = self.offsets.to(torch.long) + int_lmk
        diffs = landmark - int_lmk.to(landmark.dtype)       

        offsets_subpix = self.offsets - diffs
        vals = 1 / (1 + (offsets_subpix ** 2).sum(dim=1) + 1e-6).sqrt()

        img[0, locations[:, 0], locations[:, 1]] = vals.clone()

        return img

    def draw_landmarks(self, landmarks):


        landmarks = landmarks.view(-1, 2)

        #landmarks = landmarks.clone()

        for i in range(landmarks.size(-1)):
            landmarks[:, i] = torch.clamp(
                landmarks[:, i].clone(),
                self.half_size,
                self.img_shape[1 - i] - 1 - self.half_size)

        return torch.max(torch.cat([self.draw_lmk_helper(lmk.unsqueeze(0))
                                    for lmk in landmarks], dim=0), dim=0,
                         keepdim=True)[0]

    def forward(self, landmark_batch):

        #print(landmark_batch.size())

        return torch.cat([self.draw_landmarks(landmarks).unsqueeze(0)
                          for landmarks in landmark_batch], dim=0)