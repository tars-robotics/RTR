"""
Modified from VQ-BeT https://github.com/jayLEE0301/vq_bet_official
Some code is adapted from Stable Diffusion https://github.com/CompVis/stable-diffusion
"""
import torch.nn as nn
import math
import einops
from .vae_utils import *
import torch.nn.functional as F

def interp_coord(T_out, T_in, device=None, dtype=torch.float32):
    """
    Return interpolated coordinates for position embedding:
    T_in: original length
    T_out: length after interpolation
    Example:
    T_in = 10 → [0, 0.1, ..., 0.9]
    T_out = 20 → [0, 0.05, ..., 0.95]
    """
    return torch.linspace(
        0, 
        1 - 1/T_out, 
        T_out, 
        device=device,
        dtype=dtype
    )   # shape = (T_out,)



def interp_1d(x, T_out, T_in,  mode="linear", align_corners=True):
    """
    x: (B, T*A)
    T_out: target temporal length
    return: (B, T_out, A)
    """
    x = einops.rearrange(x, "N (T A) -> N A T", T=T_in)

    y_1d = F.interpolate(
        x,
        size=T_out,
        mode=mode,                   # "linear" corresponds to 1D linear interpolation
        align_corners=align_corners
    )                                # (B, A, T_out)

    y = y_1d.permute(0, 2, 1)        # (B, T_out, A)
    return y

def downsample_1d(x_up, T_in, mode="linear"):
    """
    x_up: (N, T_out, A)
    return: (N, T_in, A)
    """

    # (N, A, T_out)
    x = x_up.permute(0, 2, 1)

    x_down = F.interpolate(
        x,
        size=T_in,
        mode=mode,
        align_corners=True
    )  # (N, A, T_in)

    return x_down.permute(0, 2, 1)


class MLPPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coord):
        """
        coord: (T,) or (B, T)
        return: (B, T, out_dim)
        """
        if coord.dim() == 1:
            coord = coord.unsqueeze(0)  # (1, T)

        B, T = coord.shape

        coord = coord.unsqueeze(-1)  # (B, T, 1)
        coord = self.net(coord)      # (B, T, out_dim)
        return coord



class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        """
        x: (B, T_out, A)
        coord: (T_out,) or (B, T_out)  # Both are supported
        """
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state

class EncoderCNN(nn.Module):
    def __init__(self,
                    input_dim,
                    output_dim=16,
                    hidden_dim=128,
                    layer_num=2
                    # time_compression_ratio=4
                    ):
        super(EncoderCNN, self).__init__()

        self.action_dim = input_dim
        # layer_num = int(math.log2(time_compression_ratio))

        layers = []
        for i in range(layer_num):
            if i == 0:
                # Change stride_size to avoid temporal compression.
                layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2))
            else:
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=1, padding=2))

        self.encoder = nn.Sequential(*layers)
        self.apply(weights_init_encoder)

    def forward(self, x, flatten=False):
        x = einops.rearrange(x, "N (T A) -> N T A", A=self.action_dim)
        x = einops.rearrange(x, "N T A -> N A T")
        h = self.encoder(x)
        h = einops.rearrange(h, "N C T -> N T C")
        if flatten:
            h = einops.rearrange(h, "N T C -> N (T C)")
        return h


class ActionVAE(nn.Module):
    def __init__(
        self,
        input_dim=9,
        horizon=10, # length of action chunk
        latent_dim=512,
        hidden_state=512,
        n_embed=32,
        mlp_layer_num=1,
        # time_compression_ratio=4,
        kl_multiplier=1e-6,
        device="cuda",
        load_dir=None,
        act_scale=1.0,
        T_up_ratio:int=5
    ):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        self.n_embed = n_embed
        self.kl_multiplier = kl_multiplier
        self.device = device
        self.act_scale = act_scale

        # time_compression_ratio=1#time_compression_ratio
        self.encoder = EncoderCNN(
            input_dim=self.input_dim, output_dim=latent_dim, hidden_dim=hidden_state,layer_num=mlp_layer_num
        )

        output_shape = get_output_shape((self.input_dim * self.horizon,), self.encoder)
        if len(output_shape) == 1:
            raise NotImplementedError("implicit need temporal dim")
            # decoder_n_latent_dims = output_shape[0]
            # self.downsampled_input_h = 1
        else:
            decoder_n_latent_dims =  (output_shape[0]*T_up_ratio) * (output_shape[1]) # +1 for position embedding #np.multiply(*output_shape)
            self.downsampled_input_h = output_shape[0]
        
        self.T_up = self.downsampled_input_h*T_up_ratio
        self.T_up_ratio = T_up_ratio

        self.decoder_n_latent_dims = decoder_n_latent_dims

        self.decoder = MLP(
            input_dim=decoder_n_latent_dims, output_dim=self.input_dim * self.horizon * T_up_ratio, layer_num=mlp_layer_num
        )
        self.latent_dim = latent_dim

        self.mlp_position_encoder = MLPPositionalEncoding(hidden_dim=hidden_state, out_dim=output_shape[1])  # Align with decoder.

        self.quant = torch.nn.Conv1d(self.latent_dim, 2*self.n_embed, 1)
        self.post_quant = torch.nn.Conv1d(self.n_embed, self.latent_dim, 1)
        self.embedding_dim = self.latent_dim

        self.optim_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )

        self.optim_params += list(self.quant.parameters())
        self.optim_params += list(self.post_quant.parameters())

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

    def get_action_from_latent(self, latent):
        output = self.decoder(latent) * self.act_scale
        if self.horizon == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.horizon == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def quant_state(self, state):
        batch_size = state.size(0)
        if len(state.shape) == 2:
            state = einops.rearrange(state, "N (T A) -> N A T", T=self.downsampled_input_h)
        else:
            state = einops.rearrange(state, "N T A -> N A T")

        moments = self.quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        state_vq = posterior.sample()
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq, posterior

    def postprocess_quant_state(self, state_vq):# return N A T
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq
    
    def decode_with_pos(self, x, coord):
        """
        decoder: MLP decoder
        x: (B, T_out, A)
        coord: (T_out,) or (1, T_out)
        return: (B, T_out*A)
        """

        B = x.size(0)

        # # coord: (T_out,) → (B, T_out, 1)
        # if coord.dim() == 1:
        #     coord = coord.unsqueeze(0).expand(B, -1).unsqueeze(-1)
        # else:
        #     coord = coord.unsqueeze(-1)

        # # Concatenate the position dimension.
        # x_with_pos = torch.cat([x, coord], dim=-1)   # (B, T_out, A+1)
        # import pdb;pdb.set_trace()
        pe = self.mlp_position_encoder(coord) # B, T, outdim
        x_with_pos = x + pe # B, T, outdim

        # flatten
        x_flat = x_with_pos.reshape(B, -1)           # (B, T_out*(A+1))

        # Feed into MLP.
        out = self.decoder(x_flat)                        # (B, T_out*A)

        return out

    def calculate_loss(self, state,use_cos=False):
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state.float())

        state_vq, posterior = self.quant_state(state_rep)
        
        state_vq = self.postprocess_quant_state(state_vq)# N (T A)
        
        # interpolate
        state_vq = interp_1d(state_vq, T_out=self.T_up, T_in=self.downsampled_input_h) # N T A
        coord = interp_coord(self.T_up, self.downsampled_input_h, state_vq.device)

        state = interp_1d(state, T_out=self.T_up, T_in=self.downsampled_input_h) # N T A
        state = einops.rearrange(state, "N T A -> N (T A)")

        dec_out = self.decode_with_pos(state_vq, coord) # N (T A)
        # dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        kl_loss = posterior.kl().mean() * self.kl_multiplier
        if use_cos:
            dec_out = dec_out.reshape(dec_out.shape[0], self.T_up, -1)# B, T, A
            # TODO: add smooth loss
            d1 = dec_out[:, 1:, :] - dec_out[:, :-1, :]         # (B, T-1, A)
            d2 = d1[:, 1:, :]                                   # (B, T-2, A)
            d1_trim = d1[:, :-1, :]                             # (B, T-2, A)
            cos = torch.nn.functional.cosine_similarity(d1_trim, d2, dim=-1)  # (B, T-2)
            # smooth_loss = 1 - cos.mean()
            smooth_loss = (1 - cos).mean() * 0.001
            loss = encoder_loss + kl_loss + smooth_loss

        else:
            loss = encoder_loss + kl_loss
            smooth_loss = torch.zeros_like(encoder_loss)

        return loss, encoder_loss, kl_loss, smooth_loss
    
    def encode_then_decode(self, state):
        """
        state should be torch.Tensor(device) which is normalized
        """
        # batch = normalizer[action_space].normalize(batch)
        # batch = batch.to(device)
        action_dim = state.shape[-1]
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state.float())

        state_vq, posterior = self.quant_state(state_rep)
        
        state_vq = self.postprocess_quant_state(state_vq)# N (T A)
        
        # interpolate
        state_vq = interp_1d(state_vq, T_out=self.T_up, T_in=self.downsampled_input_h) # N T A
        coord = interp_coord(self.T_up, self.downsampled_input_h, state_vq.device)

        state = interp_1d(state, T_out=self.T_up, T_in=self.downsampled_input_h) # N T A
        state = einops.rearrange(state, "N T A -> N (T A)")

        dec_out = self.decode_with_pos(state_vq, coord) # N (T A)

        dec_out = dec_out * self.act_scale
        dec_out = dec_out.view(dec_out.shape[0], -1, action_dim) # N T_out A

        # TODO: derive this from up_.
        # indices = torch.arange(self.downsampled_input_h,)
        dec_out = downsample_1d(dec_out, self.downsampled_input_h)

        # dec_out = normalizer[action_space].unnormalize(dec_out)# device may be incorrect

        return dec_out


    def state_dict(self):
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        state_dict["quant"] = self.quant.state_dict()
        state_dict["post_quant"] = self.post_quant.state_dict()
        state_dict["mlp_position_encoder"] = self.mlp_position_encoder.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # for compatibility
        if 'state_dicts' in state_dict:
            state_dict = state_dict['state_dicts']['model']
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.quant.load_state_dict(state_dict["quant"])
        self.post_quant.load_state_dict(state_dict["post_quant"])
        self.mlp_position_encoder.load_state_dict(state_dict["mlp_position_encoder"])
