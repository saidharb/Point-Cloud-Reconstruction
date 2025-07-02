import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from models.DeepCAD.model.autoencoder import Bottleneck, Decoder
from models.DeepCAD.cadlib.macro import MAX_TOTAL_LEN, ARGS_DIM, ALL_COMMANDS, N_ARGS

def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)

class Config():
    def __init__(self):
        self.d_model = 256
        self.dim_z = 256
        self.max_total_len = MAX_TOTAL_LEN
        self.n_heads = 8
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.n_layers_decode = 4
        self.args_dim = ARGS_DIM
        self.n_commands = len(ALL_COMMANDS)  
        self.n_args = N_ARGS
        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }


class get_pn2_deepcad_model(nn.Module):
    def __init__(self, cfg, normal_channel=False):
        super(get_pn2_deepcad_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1], [64], in_channel,[[32, 32, 64]], bias=False)
        self.sa2 = PointNetSetAbstractionMsg(256, [0.2], [64], 64,[[64, 64, 128]], bias=False)
        self.sa3 = PointNetSetAbstractionMsg(128, [0.4], [64], 128,[[128, 128, 256]], bias=False)
        self.sa4 = PointNetSetAbstraction(None, None, None, 256+3, [256, 512, 1024], True, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cfg.d_model)
        self.tanh = nn.Tanh()

        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.view(B, 1024)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        x = self.bottleneck(x)
        out_logits = self.decoder(x)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        return res

 

    
    
