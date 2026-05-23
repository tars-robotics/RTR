"""
Modified from VQ-BeT https://github.com/jayLEE0301/vq_bet_official
Some code is adapted from Stable Diffusion https://github.com/CompVis/stable-diffusion
"""
import torch.nn
import einops
from ....reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from ....reactive_diffusion_policy.model.common.shape_util import get_output_shape
from ....reactive_diffusion_policy.model.vae.vector_quantize_pytorch.residual_vq import ResidualVQ
from ....reactive_diffusion_policy.model.vae.distributions import DiagonalGaussianDistribution
from ....reactive_diffusion_policy.model.vae.utils import *

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import math

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
                    layer_num=1):
        super(EncoderCNN, self).__init__()

        self.action_dim = input_dim

        layers = []
        for i in range(layer_num):
            if i == 0:
                layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            else:
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
        # layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2))
        if layer_num > 0:
            layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2))
        else:
            if layer_num == 0:
                layers.append(nn.Conv1d(input_dim, output_dim, kernel_size=5, stride=2, padding=2))
            elif layer_num == -1:
                layers.append(nn.Conv1d(input_dim, output_dim, kernel_size=5, stride=1, padding=2))
            else:
                raise NotImplementedError()

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

class DecoderRNN(nn.Module):
    def __init__(
        self,
        global_cond_dim,
        temporal_cond_dim,
        output_dim,
        hidden_dim,
        layer_num=1,
    ):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.GRU(global_cond_dim + temporal_cond_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_encoder)

    def forward(self, global_cond, temporal_cond):
        global_cond = global_cond.unsqueeze(1).expand(-1, temporal_cond.size(1), -1)
        x = torch.cat([global_cond, temporal_cond], dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = einops.rearrange(x, "N T A -> N (T A)")
        return x

class DecoderCNN(nn.Module):
    """
    ConvTranspose1d-based temporal upsampling decoder.
    Input:  latent_flat: (N, downsampled_T * latent_C)
    Output: action_flat: (N, horizon * action_dim)
    """
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        downsampled_T: int,
        latent_C: int,          # usually n_latent_dims
        hidden_dim: int = 512,
        upsample_layers: int = 2,   # number of stride-2 upsampling blocks
        kernel_size: int = 5,
        use_fc_refine: bool = True,
        fc_hidden_dim: int = 1024,
        fc_layers: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.downsampled_T = downsampled_T
        self.latent_C = latent_C
        self.use_fc_refine = use_fc_refine

        k = kernel_size
        p = k // 2  # keep "same-ish" padding behavior

        # Project channels up (optional but often helps capacity)
        self.in_proj = nn.Conv1d(latent_C, hidden_dim, kernel_size=1)

        blocks = []
        in_ch = hidden_dim
        for _ in range(upsample_layers):
            # stride=2 doubles T approximately.
            # output_padding=1 helps exact doubling for odd/even lengths.
            blocks.append(nn.ConvTranspose1d(in_ch, hidden_dim, kernel_size=k, stride=2, padding=p, output_padding=1))
            blocks.append(nn.ReLU())
            in_ch = hidden_dim

        self.upsampler = nn.Sequential(*blocks)

        # Map to action_dim channels (per-timestep action vector)
        self.out_proj = nn.Conv1d(hidden_dim, action_dim, kernel_size=1)

        if use_fc_refine:
            mlp = []
            in_dim = horizon * action_dim
            mlp.append(nn.Linear(in_dim, fc_hidden_dim))
            mlp.append(nn.ReLU())
            for _ in range(fc_layers):
                mlp.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(fc_hidden_dim, in_dim))
            self.fc_refine = nn.Sequential(*mlp)
        else:
            self.fc_refine = None

        # Reuse the existing initializer; replace it if a separate decoder initializer is needed.
        self.apply(weights_init_encoder)

    def forward(self, latent_flat: torch.Tensor) -> torch.Tensor:
        # latent_flat: (N, downsampled_T * latent_C)
        x = einops.rearrange(latent_flat, "N (T C) -> N C T", T=self.downsampled_T, C=self.latent_C)

        x = self.in_proj(x)          # (N, hidden_dim, T)
        x = self.upsampler(x)        # (N, hidden_dim, ~T_up)
        x = self.out_proj(x)         # (N, action_dim, ~T_up)

        # Align to horizon by truncating or padding with zeros.
        T_up = x.shape[-1]
        if T_up > self.horizon:
            x = x[..., : self.horizon]
        elif T_up < self.horizon:
            x = F.pad(x, (0, self.horizon - T_up))

        out = einops.rearrange(x, "N A T -> N (T A)")  # flatten to (N, horizon*action_dim)

        if self.fc_refine is not None:
            # Residual-style refinement is usually more stable.
            out = out + self.fc_refine(out)

        return out


class VAE:
    def __init__(
        self,
        horizon=10, # length of action chunk
        shape_meta={},
        n_latent_dims=512,
        mlp_layer_num=1,
        use_conv_encoder=False,
        conv_latent_dims=512,
        conv_layer_num=1,
        use_rnn_decoder=False,
        rnn_latent_dims=512,
        rnn_layer_num=1,
        use_vq=False,
        n_embed=32,
        vqvae_groups=4,
        kl_multiplier=1e-6,
        eval=True,
        device="cuda",
        load_dir=None,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
        use_conv_decoder=False,
        second_stage=False
    ):
        self.input_dim_h = horizon
        self.input_dim_w = shape_meta['action']['shape'][0]
        self.use_conv_encoder = use_conv_encoder
        self.use_rnn_decoder = use_rnn_decoder
        if self.use_rnn_decoder:
            all_extented_obs_keys = list(shape_meta['extended_obs'].keys())
            self.extented_obs_keys = sorted(all_extented_obs_keys)
            self.rnn_temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][0] for extented_obs_key in self.extented_obs_keys])
        self.use_vq = use_vq
        self.n_embed = n_embed
        self.vqvae_groups = vqvae_groups
        self.kl_multiplier = kl_multiplier
        self.device = device
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        self.normalizer = LinearNormalizer()

        if self.use_conv_encoder:
            self.encoder = EncoderCNN(
                input_dim=self.input_dim_w, output_dim=n_latent_dims, hidden_dim=conv_latent_dims, layer_num=conv_layer_num
            ).to(self.device)
        else:
            self.encoder = MLP(
                input_dim=self.input_dim_w * self.input_dim_h, output_dim=n_latent_dims, layer_num=mlp_layer_num
            ).to(self.device)

        output_shape = get_output_shape((self.input_dim_w * self.input_dim_h,), self.encoder)
        if len(output_shape) == 1:
            decoder_n_latent_dims = output_shape[0]
            self.downsampled_input_h = 1
        else:
            decoder_n_latent_dims = np.multiply(*output_shape)
            self.downsampled_input_h = output_shape[0]

        if self.use_rnn_decoder:
            self.decoder = DecoderRNN(global_cond_dim=decoder_n_latent_dims, temporal_cond_dim=self.rnn_temporal_cond_dim,
                                      output_dim=self.input_dim_w, hidden_dim=rnn_latent_dims,
                                      layer_num=rnn_layer_num).to(self.device)
        else:
            if not use_conv_decoder:
                self.decoder = MLP(
                    input_dim=decoder_n_latent_dims, output_dim=self.input_dim_w * self.input_dim_h, layer_num=mlp_layer_num
                ).to(self.device)
            else:
                up_layers = conv_layer_num + 1 if self.use_conv_encoder else max(1, int(math.ceil(math.log2(self.input_dim_h / max(1, self.downsampled_input_h)))))
                self.decoder = DecoderCNN(
                    action_dim=self.input_dim_w,
                    horizon=self.input_dim_h,
                    downsampled_T=self.downsampled_input_h,
                    latent_C=self.n_latent_dims if hasattr(self, "n_latent_dims") else n_latent_dims,
                    hidden_dim=conv_latent_dims,      # Reuse the existing parameter name; add conv_decoder_hidden_dim if needed.
                    upsample_layers=up_layers,
                    use_fc_refine=True,
                    fc_hidden_dim=1024,
                    fc_layers=1,
                ).to(self.device)
        self.n_latent_dims = n_latent_dims

        if self.use_vq:
            self.vq_layer = ResidualVQ(
                dim=self.n_latent_dims,
                num_quantizers=self.vqvae_groups,
                codebook_size=self.n_embed,
            ).to(self.device)
            self.vq_layer.device = device
        else:
            self.quant = torch.nn.Conv1d(self.n_latent_dims, 2*self.n_embed, 1).to(self.device)
            self.post_quant = torch.nn.Conv1d(self.n_embed, self.n_latent_dims, 1).to(self.device)
        self.embedding_dim = self.n_latent_dims

        self.optim_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        if self.use_vq:
            self.optim_params += list(self.vq_layer.parameters())
        else:
            self.optim_params += list(self.quant.parameters())
            self.optim_params += list(self.post_quant.parameters())

        if load_dir is not None:
            try:
                print("="*100)
                print(f"load_dir of vae is {load_dir}")
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

        if self.use_vq:
            if eval:
                self.vq_layer.eval()
            else:
                self.vq_layer.train()
        self.second_stage = second_stage

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_action_from_latent(self, latent):
        output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def get_action_from_latent_with_temporal_cond(self, latent, temporal_cond):
        output = self.decoder(latent, temporal_cond) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def quant_state_with_vq(self, state):
        batch_size = state.size(0)
        if len(state.shape) == 2:
            state = einops.rearrange(state, "N (T A) -> N T A", T=self.downsampled_input_h)

        state_vq, vq_code, vq_loss_state = self.vq_layer(state)
        state_vq = state_vq.reshape(batch_size, -1)
        vq_code = vq_code.reshape(batch_size, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        return state_vq, vq_code, vq_loss_state

    def quant_state_without_vq(self, state):
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

    def postprocess_quant_state_without_vq(self, state_vq):
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq

    def get_temporal_cond(self, extended_obs_dict, extended_obs_last_step=None, extend_obs_pad_after_n=None):
        temporal_cond = []
        for extented_obs_key in self.extented_obs_keys:
            if extended_obs_last_step is not None:
                extented_obs = extended_obs_dict[extented_obs_key][..., -extended_obs_last_step:, :]
            else:
                extented_obs = extended_obs_dict[extented_obs_key]
            if extend_obs_pad_after_n is not None:
                padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                extented_obs = torch.cat([padding_obs, extented_obs], dim=-2)
            extented_obs = self.normalizer[extented_obs_key].normalize(extented_obs)
            temporal_cond.append(extented_obs)
        temporal_cond = torch.cat(temporal_cond, dim=-1)
        return temporal_cond

    def compute_loss_and_metric(self, batch):

        state = batch["action"]
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)

        return_dict = {
            "loss": encoder_loss,
            "encoder_loss": encoder_loss.clone().detach().cpu().numpy(),
            "vae_recon_loss": vae_recon_loss.item(),
        }

        if self.use_vq:
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "vq_code": vq_code,
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
                "vq_loss_state": vq_loss_state.clone().detach().cpu().numpy(),
            })
        else:
            kl_loss = posterior.kl().mean()
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (kl_loss * self.kl_multiplier)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "kl_loss": kl_loss.clone().detach().cpu().numpy(),
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
            })

        return return_dict
    
    def encode_to_latent(self, batch, normalize=True):
        """
        input: N,T,A
        output: N,T,A
        """
        if isinstance(batch, dict):
            state = batch["action"]
        else:
            state = batch
        if normalize:
            state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            # raise NotImplementedError()
            state_vq, _, _ = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
        state_vq = einops.rearrange(state_vq, 'N (T A) -> N T A', T=self.downsampled_input_h)
        return state_vq
    
    def decode_from_latent(self, action, denormalize=True):
        """
        input: N,T(compressed),A
        output: N,T,A
        if denormalize, tensor will be moved to cpu because of normalizer
        """
        N,compress_T,A = action.shape
        action = einops.rearrange(action, 'N T A -> N (T A)')
        if self.use_vq:
            # raise NotImplementedError()
            state_vq = action
        else:
            state_vq = self.postprocess_quant_state_without_vq(action)
        
        if self.use_rnn_decoder:
            raise NotImplementedError()
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h)
        dec_out = dec_out * self.act_scale
        if denormalize:
            dec_out = self.normalizer['action'].unnormalize(dec_out)

        return dec_out

            

    def encode_then_decode(self, batch):

        state = batch["action"]
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            # Split latent policy output here
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h)
        dec_out = dec_out * self.act_scale
        dec_out = self.normalizer['action'].unnormalize(dec_out)

        return dec_out
    

    

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        if self.use_vq:
            self.vq_layer.eval()
        else:
            self.quant.eval()
            self.post_quant.eval()

    def train(self):
        if not self.second_stage:
            self.encoder.train()
            self.decoder.train()
            if self.use_vq:
                self.vq_layer.train()
            else:
                self.quant.train()
                self.post_quant.train()
        else:
            self.decoder.train()
            if self.use_vq:
                # self.vq_layer.train()
                raise NotImplementedError("")# If using vq, it's necessary to determine whether vq_layer is called before or after the latent; this is not currently supported.
            else:
                self.post_quant.train()

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        if self.use_vq:
            self.vq_layer.to(device)
        else:
            self.quant.to(device)
            self.post_quant.to(device)
        self.device = device

    def state_dict(self):
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "normalizer": self.normalizer.state_dict()
        }
        if self.use_vq:
            state_dict["vq_embedding"] = self.vq_layer.state_dict()
        else:
            state_dict["quant"] = self.quant.state_dict()
            state_dict["post_quant"] = self.post_quant.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # for compatibility
        if 'state_dicts' in state_dict:
            state_dict = state_dict['state_dicts']['model']
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        if "normalizer" in state_dict.keys():
            self.normalizer.load_state_dict(state_dict["normalizer"])
        else:
            print(f"normalizer not in state_dict.keys() in load_state_dict of vae")
        if self.use_vq:
            self.vq_layer.load_state_dict(state_dict["vq_embedding"])
            self.vq_layer.eval()
        else:
            self.quant.load_state_dict(state_dict["quant"])
            self.post_quant.load_state_dict(state_dict["post_quant"])

    def _load_latent_dataset_statistics(self, path: Union[str, Path]=None, dataset_statistics:Dict = None) -> None:
        if dataset_statistics == None:
            path = Path(path)
            with path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = dataset_statistics

        self._dataset_stats = stats   # contains action/proprio/...
    
    def _get_action_stats_torch(self, device, dtype, is_latent=False, is_proprio=False):
        if is_proprio:
            raise NotImplementedError("proprio stats not provided in this codebase")
        if is_latent:
            s = self._dataset_stats["latent_action"]  
        else:
            s = self._dataset_stats["action"]

        def t(x):
            return torch.tensor(x, device=device, dtype=dtype)

        mn  = t(s["min"])
        mx  = t(s["max"])
        q01 = t(s["q01"])
        q99 = t(s["q99"])

        # Optional fields:mean/std(read them if present; QUANTILES does not require them)
        mean = t(s["mean"]) if "mean" in s else torch.zeros_like(q01)
        std  = t(s["std"])  if "std"  in s else torch.ones_like(q01)

        if "mask" in s:
            mask = torch.tensor(s["mask"], device=device, dtype=torch.bool)
        else:
            mask = torch.ones_like(q01, dtype=torch.bool, device=device)

        # zeros_mask:force normalized values to 0 when quantiles or min/max have zero range
        zeros_mask = (q01 == q99) | (mn == mx)

        # reshape for broadcasting to [N,T,A]
        def v(x): return x.view(1, 1, -1)
        return dict(
            mean=v(mean), std=v(std),
            min=v(mn), max=v(mx),
            q01=v(q01), q99=v(q99),
            mask=v(mask),
            zeros_mask=v(zeros_mask),
        )

    def normalize_from_dataset(self, action: torch.Tensor, normalization_type="NORMAL", is_latent=True, is_proprio=False) -> torch.Tensor:
        """
        Input:  [N,T,A] torch.Tensor
        Output:  [N,T,A] torch.Tensor
        NORMAL has higher precise
        """
        assert action.ndim == 3, f"Expected [N,T,A], got {tuple(action.shape)}"
        if normalization_type is None:
            normalization_type = self.action_proprio_normalization_type

        st = self._get_action_stats_torch(device=action.device, dtype=action.dtype,is_latent=is_latent,is_proprio=is_proprio)

        if normalization_type == "NORMAL":
            # tf.where(mask, (x-mean)/std, x)
            out = torch.where(st["mask"], (action - st["mean"]) / (st["std"] + 1e-8), action)
            return out

        elif normalization_type == "QUANTILES":
            low, high = st["q01"], st["q99"]

            # tf.where(mask, clip(2*(x-low)/(high-low)-1, -1, 1), x)
            scaled = 2 * (action - low) / (high - low + 1e-8) - 1.0
            scaled = torch.clamp(scaled, -1.0, 1.0)
            out = torch.where(st["mask"], scaled, action)

            # zeros_mask: min==max -> set to 0.0 (matches TF behavior and is not affected by mask)
            out = torch.where(st["zeros_mask"], torch.zeros_like(out), out)
            return out

        raise ValueError(f"Unknown Normalization Type {normalization_type}")

    def denormalize_from_dataset(self, action: torch.Tensor, normalization_type="NORMAL", is_latent=True,is_proprio=False) -> torch.Tensor:
        """
        Input:  [N,T,A] torch.Tensor (normalized space)
        Output:  [N,T,A] torch.Tensor (original action space)
        Note: QUANTILES normalization clips values; unnormalization can only recover the clipped interval mapping, not clipped-away information.
        """
        assert action.ndim == 3, f"Expected [N,T,A], got {tuple(action.shape)}"
        if normalization_type is None:
            normalization_type = self.action_proprio_normalization_type

        st = self._get_action_stats_torch(device=action.device, dtype=action.dtype, is_latent=is_latent,is_proprio=is_proprio)

        if normalization_type == "NORMAL":
            # inverse: x = x*std + mean  (only masked dims)
            out = torch.where(st["mask"], action * (st["std"] + 1e-8) + st["mean"], action)
            return out

        elif normalization_type in "QUANTILES":
            low, high = st["q01"], st["q99"]

            # inverse of y = 2*(x-low)/(high-low)-1  => x = (y+1)/2*(high-low)+low
            inv = (action + 1.0) * 0.5 * (high - low) + low
            out = torch.where(st["mask"], inv, action)

            # zeros_mask dims are constants in original space (min==max), restore them to that constant
            out = torch.where(st["zeros_mask"], st["min"], out)
            return out

        raise ValueError(f"Unknown Normalization Type {normalization_type}")

    # two-stage
    def set_requires_grad(self, module: torch.nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def second_stage_train(self):
        print(f"must _load_dataset_statistics which contains latent statistics before two-stage-training!!!")

        ## freeze for encoder and quant
        self.set_requires_grad(self.encoder, False)
        self.encoder.eval()  # freeze if dropout/BN is present
        self.set_requires_grad(self.quant, False)
        self.quant.eval()  # freeze if dropout/BN is present

        ## optim, only for decoder and post_quant
        self.optim_params = (
            list(self.decoder.parameters())
        )
        if self.use_vq:
            self.optim_params += list(self.vq_layer.parameters())
        else:
            self.optim_params += list(self.post_quant.parameters())
    
    def compute_loss_and_metric_second_stage(self, batch, vla_action_latent=None):
        if isinstance(batch,dict):
            state = batch["action"]
        else:
            state = batch
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            if vla_action_latent is None:
                raise ValueError("vla_action_latent should not be None")
            else:
                state_vq = self.denormalize_from_dataset(vla_action_latent, is_latent=True)
                state_vq = einops.rearrange(state_vq, 'N T A -> N (T A)')
                state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            raise NotImplementedError("")
        else:
            dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)

        return_dict = {
            "loss": encoder_loss,
            "encoder_loss": encoder_loss.clone().detach().cpu().numpy(),
            "vae_recon_loss": vae_recon_loss.item(),
        }

        if self.use_vq:
            raise NotImplementedError("")
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "vq_code": vq_code,
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
                "vq_loss_state": vq_loss_state.clone().detach().cpu().numpy(),
            })
        else:
            # kl_loss = posterior.kl().mean()
            rep_loss = encoder_loss * self.encoder_loss_multiplier #+ (kl_loss * self.kl_multiplier)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "kl_loss": rep_loss.clone().detach().cpu().numpy(),
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
            })

        return return_dict


class Openvla_oft_VAE(VAE):
    """
    This class does not own a normalizer; training uses the normalizer from the OpenVLA-OFT dataset, so standalone open-loop testing is unavailable until the OpenVLA dataset is integrated.
    """
    def __init__(
        self,
        normalize_latent,
        second_stage=False, 
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize_latent = normalize_latent
        self.second_stage = second_stage
        # if second_stage:
            # now use hardcore for convinient
        self.min_action = -1
        self.max_action = 1
        self.n_bins = 256
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def train(self):
        if self.second_stage:
            self.decoder.train()
            if self.use_vq:
                # self.vq_layer.train()
                raise NotImplementedError("")# Need to decide whether vq_layer runs before or after latent
            else:
                self.post_quant.train()
        else:
            self.encoder.train()
            self.decoder.train()
            if self.use_vq:
                self.vq_layer.train()
            else:
                self.quant.train()
                self.post_quant.train()
    
    def set_requires_grad(self, module: torch.nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def second_stage_train(self):
        print(f"must _load_dataset_statistics which contains latent statistics before two-stage-training!!!")

        ## freeze for encoder and quant
        self.set_requires_grad(self.encoder, False)
        self.encoder.eval()  # freeze if dropout/BN is present
        self.set_requires_grad(self.quant, False)
        self.quant.eval()  # freeze if dropout/BN is present

        ## optim, only for decoder and post_quant
        self.optim_params = (
            list(self.decoder.parameters())
        )
        if self.use_vq:
            self.optim_params += list(self.vq_layer.parameters())
        else:
            self.optim_params += list(self.post_quant.parameters())

    def compute_loss_and_metric_second_stage(self, batch, vla_action_latent=None):
        """
        action_quant aware training
        """
        if isinstance(batch,dict):
            state = batch["action"]
        else:
            state = batch
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            if vla_action_latent == None:
                state_vq, posterior = self.quant_state_without_vq(state_rep)
                state_vq_device = state_vq.device
                state_vq_dtype = state_vq.dtype
                # normalize then action_quant
                # import pdb;pdb.set_trace()
                state_vq = einops.rearrange(state_vq, 'N (T A) -> N T A', T=self.downsampled_input_h)
                # Normalize before feeding into VLA
                state_vq = self.normalize_from_dataset(state_vq, is_latent=True)
                # print(f"before action_quant, state_vq[0,0:2] is {state_vq[0,0:2]}")
                state_vq = state_vq.cpu().numpy()
                # Convert to tokens with action_tokenizer
                state_vq = np.clip(state_vq, a_min=float(self.min_action), a_max=float(self.max_action))
                state_vq = np.digitize(state_vq, self.bins)

                # action_dequant then unnormalize
                # Detokenize action_tokenizer output to actions
                state_vq = np.clip(state_vq - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
                state_vq = self.bin_centers[state_vq]
                state_vq = torch.tensor(state_vq).to(state_vq_device,dtype=state_vq_dtype)
                # print(f"after action_quant, state_vq[0,0:2] is {state_vq[0,0:2]}")
                # During prediction, denormalize to latent space and decode actions with vae.decode
                state_vq = self.denormalize_from_dataset(state_vq, is_latent=True)
                state_vq = einops.rearrange(state_vq, 'N T A -> N (T A)')
                # only post_quant and decoder need to be update
                state_vq = self.postprocess_quant_state_without_vq(state_vq)
            else:
                state_vq = self.denormalize_from_dataset(vla_action_latent, is_latent=True)
                state_vq = einops.rearrange(state_vq, 'N T A -> N (T A)')
                # only post_quant and decoder need to be update
                state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)

        return_dict = {
            "loss": encoder_loss,
            "encoder_loss": encoder_loss.clone().detach().cpu().numpy(),
            "vae_recon_loss": vae_recon_loss.item(),
        }

        if self.use_vq:
            raise NotImplementedError("")
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "vq_code": vq_code,
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
                "vq_loss_state": vq_loss_state.clone().detach().cpu().numpy(),
            })
        else:
            # kl_loss = posterior.kl().mean()
            rep_loss = encoder_loss * self.encoder_loss_multiplier #+ (kl_loss * self.kl_multiplier)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "kl_loss": rep_loss.clone().detach().cpu().numpy(),
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
            })

        return return_dict


    def state_dict(self):
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        if self.use_vq:
            state_dict["vq_embedding"] = self.vq_layer.state_dict()
        else:
            state_dict["quant"] = self.quant.state_dict()
            state_dict["post_quant"] = self.post_quant.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # for compatibility
        if 'state_dicts' in state_dict:
            state_dict = state_dict['state_dicts']['model']
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        if self.use_vq:
            self.vq_layer.load_state_dict(state_dict["vq_embedding"])
            self.vq_layer.eval()
        else:
            self.quant.load_state_dict(state_dict["quant"])
            self.post_quant.load_state_dict(state_dict["post_quant"])
    
    def compute_loss_and_metric(self, batch):
        state = batch["action"]
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)

        return_dict = {
            "loss": encoder_loss,
            "encoder_loss": encoder_loss.clone().detach().cpu().numpy(),
            "vae_recon_loss": vae_recon_loss.item(),
        }

        if self.use_vq:
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "vq_code": vq_code,
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
                "vq_loss_state": vq_loss_state.clone().detach().cpu().numpy(),
            })
        else:
            kl_loss = posterior.kl().mean()
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (kl_loss * self.kl_multiplier)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "kl_loss": kl_loss.clone().detach().cpu().numpy(),
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
            })

        return return_dict

    def encode_then_decode(self, batch):
        if isinstance(batch,dict):
            state = batch["action"]
        else:
            state = batch
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            print(f"max is {torch.max(state_vq)} min is {torch.min(state_vq)}")
            # Split latent policy output here
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h)
        dec_out = dec_out * self.act_scale

        return dec_out
    
    def encode_then_decode_action_quant(self, batch):
        if isinstance(batch,dict):
            state = batch["action"]
        else:
            state = batch
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            state_vq_device = state_vq.device
            state_vq_dtype = state_vq.dtype
            # normalize then action_quant
            # import pdb;pdb.set_trace()
            state_vq = einops.rearrange(state_vq, 'N (T A) -> N T A', T=self.downsampled_input_h)
            # Normalize before feeding into VLA
            state_vq = self.normalize_from_dataset(state_vq, is_latent=True)
            # print(f"before action_quant, state_vq[0,0:2] is {state_vq[0,0:2]}")
            state_vq = state_vq.cpu().numpy()
            # Convert to tokens with action_tokenizer
            state_vq = np.clip(state_vq, a_min=float(self.min_action), a_max=float(self.max_action))
            state_vq = np.digitize(state_vq, self.bins)

            # action_dequant then unnormalize
            # Detokenize action_tokenizer output to actions
            state_vq = np.clip(state_vq - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            state_vq = self.bin_centers[state_vq]
            state_vq = torch.tensor(state_vq).to(state_vq_device,dtype=state_vq_dtype)
            # print(f"after action_quant, state_vq[0,0:2] is {state_vq[0,0:2]}")
            # During prediction, denormalize to latent space and decode actions with vae.decode
            state_vq = self.denormalize_from_dataset(state_vq, is_latent=True)
            state_vq = einops.rearrange(state_vq, 'N T A -> N (T A)')
            # only post_quant and decoder need to be update
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h)
        dec_out = dec_out * self.act_scale

        return dec_out
    

    def decode_from_latent(self, action):
        """
        input: N,T(compressed),A
        output: N,T,A
        """
        N,compress_T,A = action.shape
        action = einops.rearrange(action, 'N T A -> N (T A)')
        if self.use_vq:
            raise NotImplementedError()
        else:
            state_vq = self.postprocess_quant_state_without_vq(action)
        
        if self.use_rnn_decoder:
            raise NotImplementedError()
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h)
        dec_out = dec_out * self.act_scale

        return dec_out

    def encode_to_latent(self, batch):
        """
        input: N,T,A
        output: N,T,A
        """
        if isinstance(batch, dict):
            state = batch["action"]
        else:
            state = batch
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            raise NotImplementedError()
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
        state_vq = einops.rearrange(state_vq, 'N (T A) -> N T A', T=self.downsampled_input_h)
        return state_vq
    

    def _load_dataset_statistics(self, path: Union[str, Path]=None, dataset_key: Optional[str] = None, dataset_statistics:Dict = None) -> None:
        if dataset_statistics == None:
            path = Path(path)
            with path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = dataset_statistics

        if dataset_key is None:
            # Use the first dataset by default; the JSON structure has one top-level key.
            dataset_key = next(iter(stats.keys()))

        self._dataset_key = dataset_key
        self._dataset_stats = stats[dataset_key]   # contains action/proprio/...
        self._action_stats_cache = None
    
    def _get_action_stats_torch(self, device, dtype, is_latent=False, is_proprio=False):
        """
        Convert action stats from JSON to torch tensors and cache them, recasting by device/dtype as needed.
        """
        if is_proprio:
            s = self._dataset_stats["proprio"]
        else:
            if is_latent:
                s = self._dataset_stats["naction"]
            else:
                s = self._dataset_stats["action"]

        def t(x):
            return torch.tensor(x, device=device, dtype=dtype)

        mean = t(s["mean"])
        std  = t(s["std"])
        mn   = t(s["min"])
        mx   = t(s["max"])
        q01  = t(s["q01"])
        q99  = t(s["q99"])

        if "mask" in s:
            mask = torch.tensor(s["mask"], device=device, dtype=torch.bool)
        else:
            mask = torch.ones_like(mean, dtype=torch.bool, device=device)

        zeros_mask = (mn == mx)  # Match TF behavior.

        # Reshape for broadcasting to [N, T, A].
        mean = mean.view(1, 1, -1)
        std  = std.view(1, 1, -1)
        mn   = mn.view(1, 1, -1)
        mx   = mx.view(1, 1, -1)
        q01  = q01.view(1, 1, -1)
        q99  = q99.view(1, 1, -1)
        mask = mask.view(1, 1, -1)
        zeros_mask = zeros_mask.view(1, 1, -1)

        return dict(mean=mean, std=std, min=mn, max=mx, q01=q01, q99=q99, mask=mask, zeros_mask=zeros_mask)

    def normalize_from_dataset(self, action: torch.Tensor, normalization_type="BOUNDS_Q99", is_latent=False, is_proprio=False) -> torch.Tensor:
        """
        Input:  [N,T,A] torch.Tensor
        Output:  [N,T,A] torch.Tensor
        Match the behavior of normalize_action_and_proprio() for 'action'.
        normalization_type should match `openvla-oft/prismatic/vla/constants.py`
        """
        assert action.ndim == 3, f"Expected [N,T,A], got {tuple(action.shape)}"
        if normalization_type is None:
            normalization_type = self.action_proprio_normalization_type

        st = self._get_action_stats_torch(device=action.device, dtype=action.dtype,is_latent=is_latent,is_proprio=is_proprio)

        if normalization_type == "NORMAL":
            # tf.where(mask, (x-mean)/std, x)
            out = torch.where(st["mask"], (action - st["mean"]) / (st["std"] + 1e-8), action)
            return out

        elif normalization_type in ["BOUNDS", "BOUNDS_Q99"]:
            if normalization_type == "BOUNDS":
                low, high = st["min"], st["max"]
            else:
                low, high = st["q01"], st["q99"]

            # tf.where(mask, clip(2*(x-low)/(high-low)-1, -1, 1), x)
            scaled = 2 * (action - low) / (high - low + 1e-8) - 1.0
            scaled = torch.clamp(scaled, -1.0, 1.0)
            out = torch.where(st["mask"], scaled, action)

            # zeros_mask: min==max -> set to 0.0 (matches TF behavior and is not affected by mask)
            out = torch.where(st["zeros_mask"], torch.zeros_like(out), out)
            return out

        raise ValueError(f"Unknown Normalization Type {normalization_type}")

    def denormalize_from_dataset(self, action: torch.Tensor, normalization_type="BOUNDS_Q99", is_latent=False,is_proprio=False) -> torch.Tensor:
        """
        Input:  [N,T,A] torch.Tensor (normalized space)
        Output:  [N,T,A] torch.Tensor (original action space)
        Note: BOUNDS normalization clips values, so denormalization only recovers the clipped interval mapping and cannot recover clipped information.
        """
        assert action.ndim == 3, f"Expected [N,T,A], got {tuple(action.shape)}"
        if normalization_type is None:
            normalization_type = self.action_proprio_normalization_type

        st = self._get_action_stats_torch(device=action.device, dtype=action.dtype, is_latent=is_latent,is_proprio=is_proprio)

        if normalization_type == "NORMAL":
            # inverse: x = x*std + mean  (only masked dims)
            out = torch.where(st["mask"], action * (st["std"] + 1e-8) + st["mean"], action)
            return out

        elif normalization_type in ["BOUNDS", "BOUNDS_Q99"]:
            if normalization_type == "BOUNDS":
                low, high = st["min"], st["max"]
            else:
                low, high = st["q01"], st["q99"]

            # inverse of y = 2*(x-low)/(high-low)-1  => x = (y+1)/2*(high-low)+low
            inv = (action + 1.0) * 0.5 * (high - low) + low
            out = torch.where(st["mask"], inv, action)

            # zeros_mask dims are constants in original space (min==max), restore them to that constant
            out = torch.where(st["zeros_mask"], st["min"], out)
            return out

        raise ValueError(f"Unknown Normalization Type {normalization_type}")