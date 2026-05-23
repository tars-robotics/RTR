"""
Modified from VQ-BeT https://github.com/jayLEE0301/vq_bet_official
Some code is adapted from Stable Diffusion https://github.com/CompVis/stable-diffusion
"""
import torch.nn as nn
import math
import einops
from ....tactile_generation_policy.model.action.vae_utils import *

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=3,
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
                    time_compression_ratio=4,
                    layer_num=3
                    ):
        super(EncoderCNN, self).__init__()

        self.action_dim = input_dim
        if time_compression_ratio != 1:
            layer_num = int(math.log2(time_compression_ratio))
            

        layers = []
        if time_compression_ratio != 1:
            for i in range(layer_num):
                if i == 0:
                    layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
                else:
                    layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
                layers.append(nn.ReLU())
        else:
            for i in range(layer_num):
                if i == 0:
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
        time_compression_ratio=4,
        kl_multiplier=1e-6,
        device="cuda",
        load_dir=None,
        act_scale=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        self.n_embed = n_embed
        self.kl_multiplier = kl_multiplier
        self.device = device
        self.act_scale = act_scale

        self.encoder = EncoderCNN(
            input_dim=self.input_dim, output_dim=latent_dim, hidden_dim=hidden_state, time_compression_ratio=time_compression_ratio, 
            layer_num = mlp_layer_num
        )

        output_shape = get_output_shape((self.input_dim * self.horizon,), self.encoder)
        print(f"output_shape is {output_shape}")

        if len(output_shape) == 1:
            decoder_n_latent_dims = output_shape[0]
            self.downsampled_input_h = 1
        else:
            decoder_n_latent_dims = np.multiply(*output_shape)
            self.downsampled_input_h = output_shape[0]

        self.decoder = MLP(
            input_dim=decoder_n_latent_dims, output_dim=self.input_dim * self.horizon, layer_num=mlp_layer_num
        )
        self.latent_dim = latent_dim

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

    def postprocess_quant_state(self, state_vq):
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq

    def calculate_loss(self, state):
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state.float())

        state_vq, posterior = self.quant_state(state_rep)
        
        state_vq = self.postprocess_quant_state(state_vq)

        dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        kl_loss = posterior.kl().mean() * self.kl_multiplier

        dec_out = dec_out.reshape(dec_out.shape[0], self.horizon, -1)# B, T, A
        # TODO: add smooth loss
        d1 = dec_out[:, 1:, :] - dec_out[:, :-1, :]         # (B, T-1, A)
        d2 = d1[:, 1:, :]                                   # (B, T-2, A)
        d1_trim = d1[:, :-1, :]                             # (B, T-2, A)
        cos = torch.nn.functional.cosine_similarity(d1_trim, d2, dim=-1)  # (B, T-2)
        # smooth_loss = 1 - cos.mean()
        smooth_loss = (1 - cos).mean() * 0.01

        loss = encoder_loss + kl_loss + smooth_loss

        # print(f"debug: loss is {loss} encoder_loss is {encoder_loss}, kl_loss is {kl_loss}, smooth_loss is {smooth_loss}")

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
        
        state_vq = self.postprocess_quant_state(state_vq)

        dec_out = self.decoder(state_vq)

        dec_out = dec_out * self.act_scale
        dec_out = dec_out.view(dec_out.shape[0], -1, action_dim)
        # dec_out = normalizer[action_space].unnormalize(dec_out)# device may be incorrect

        return dec_out


    def state_dict(self):
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        state_dict["quant"] = self.quant.state_dict()
        state_dict["post_quant"] = self.post_quant.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # for compatibility
        if 'state_dicts' in state_dict:
            state_dict = state_dict['state_dicts']['model']
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.quant.load_state_dict(state_dict["quant"])
        self.post_quant.load_state_dict(state_dict["post_quant"])
