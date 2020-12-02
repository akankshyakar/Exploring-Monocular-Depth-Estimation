import torch
import torch.nn as nn
from geom_utils import *

def params_to_ranges(predicted_params, world_coords_gt, mask_gt, args):
    assert len(predicted_params.shape) == 2 and predicted_params.shape[-1] == 4

    predicted_params_sigmoid = torch.sigmoid(predicted_params[:, :3])
    focal = args.FOV_deg[0] + (args.FOV_deg[1] - args.FOV_deg[0]) * predicted_params_sigmoid[:, 0]
    pitch = args.pitch_deg[0] + (args.pitch_deg[1] - args.pitch_deg[0]) * predicted_params_sigmoid[:, 1]
    roll = args.roll_deg[0] + (args.roll_deg[1] - args.roll_deg[0]) * predicted_params_sigmoid[:, 2]
    height = predicted_params[:, 3]

    return torch.cat((focal[:, None], height[:, None], pitch[:, None], roll[:, None]), dim=1)

def prepare_dimensions_and_compute_N(prediction, gt, mask):
  if len(prediction.shape) == 3:
    prediction = prediction[:, None, :, :]
  if len(gt.shape) == 3:
    gt = gt[:, None, :, :]
  assert prediction.shape == gt.shape
  if not mask is None:
    assert prediction.shape[0] == mask.shape[0]
    assert prediction.shape[-2:] == mask.shape[-2:]
    if len(mask.shape) == 3:
      mask = mask[:, None, :, :]
    assert mask.shape[1] == 1
    N = prediction.shape[1] * mask.sum(-1).sum(-1).sum(-1)
    N[N == 0] = 1
  else:
    N = prediction.shape[1] * prediction.shape[2] * prediction.shape[3]
  return prediction, gt, mask, N

def MyRMSELoss(prediction, gt, mask=None):
  prediction, gt, mask, N = prepare_dimensions_and_compute_N(prediction, gt, mask)
  d_diff = gt - prediction
  if not mask is None:
    d_diff = d_diff * mask
  mse = (d_diff ** 2).sum(-1).sum(-1).sum(-1) / N
  rmse = torch.sqrt(mse + 1e-16)
  return rmse.mean()

class CoordsRegressionLoss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self):
        super(CoordsRegressionLoss, self).__init__()
        self.input_height = 192
        self.input_width = 256
        # self.FOV_deg


    def forward(self, predicted_params_ranges, depth, world_coords_gt, mask_gt, args):
        if args.im2pcl > 0 and args.data != 'sunrgbd':
          raise Exception("im2pcl loss should be used only with sunrgbd")
        predicted_params = params_to_ranges(predicted_params_ranges, world_coords_gt, mask_gt, args)

        predicted_focal, predicted_height, predicted_pitch, predicted_roll = predicted_params[:, 0], predicted_params[:, 1], \
                                                                    predicted_params[:, 2], predicted_params[:, 3]

        batch_size = predicted_params_ranges.shape[0]
        predicted_intrinsics = fov_x_to_intrinsic_deg(predicted_focal, torch.FloatTensor([self.input_width] * batch_size).cuda(),
                                                        torch.FloatTensor([self.input_height] * batch_size).cuda(),
                                                        return_inverse=False)
        predicted_coords = pixel2cam(depth, predicted_intrinsics)

        # at this point, predicted coords should be the same as coords_g if correct
        # rotate by roll and pitch and normalize using height
        roll_rotations = zrotation_deg_torch(-1 * predicted_roll)
        pitch_rotations = xrotation_deg_torch(predicted_pitch)

        predicted_world_rotated_coords = torch.bmm(roll_rotations, predicted_coords.reshape(batch_size, 3, -1))
        predicted_world_rotated_coords = torch.bmm(pitch_rotations, predicted_world_rotated_coords).reshape(batch_size, 3,
                                                                                                            self.input_height,
                                                                                                            self.input_width)

        zeros = torch.zeros_like(predicted_height)
        height_translation = torch.cat((zeros[:, None], predicted_height[:, None], zeros[:, None]), dim=-1)
        predicted_world_coords = predicted_world_rotated_coords - height_translation[:, :, None, None]

        # mask if prediciton is too high, which can be caused by some normals
        mask_gt = mask_gt * (torch.abs(predicted_world_coords).sum(1) < 100).float()

        coords_regression_loss = MyRMSELoss(predicted_world_coords, world_coords_gt, mask_gt)
        return coords_regression_loss