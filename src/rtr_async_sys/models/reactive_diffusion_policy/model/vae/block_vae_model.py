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
        layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2))

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

class BlockEncodeVAE:
    def __init__(
        self,
        horizon=48,
        encode_horizon=12, # length of action chunk
        encode_block_num=4,# encode_horizon*encode_block_num -> horizon
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
    ):
        self.horizon = horizon
        self.encode_block_num = encode_block_num
        self.input_dim_h = encode_horizon
        self.input_dim_h_decode = encode_horizon*encode_block_num
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
            self.decoder = DecoderRNN(global_cond_dim=decoder_n_latent_dims*self.encode_block_num, temporal_cond_dim=self.rnn_temporal_cond_dim,
                                      output_dim=self.input_dim_w, hidden_dim=rnn_latent_dims,
                                      layer_num=rnn_layer_num).to(self.device)
        else:
            self.decoder = MLP(
                input_dim=decoder_n_latent_dims*self.encode_block_num, output_dim=self.input_dim_w*self.input_dim_h_decode, layer_num=mlp_layer_num
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
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

        if self.use_vq:
            if eval:
                self.vq_layer.eval()
            else:
                self.vq_layer.train()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

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
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h*self.encode_block_num)
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
        batch_size, t_encode, action_dim = state.shape
        state = state.reshape(batch_size*self.encode_block_num,t_encode//self.encode_block_num, action_dim)
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)  # Keep posterior unchanged because T/batch reshaping may affect it.
            state_vq = einops.rearrange(state_vq, "N (T A) -> N T A", T=self.downsampled_input_h).reshape(batch_size, self.downsampled_input_h*self.encode_block_num, -1).reshape(batch_size,-1)
            # import pdb;pdb.set_trace()
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        state = state.reshape(batch_size, t_encode, action_dim).reshape(batch_size,-1)
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


    def encode_to_latent(self, batch, reshape_in_vae=True):
        """
        If reshape_in_vae is False, callers must reshape before and after encode_to_latent.
        before use encode_to_latent, reshape action from (N,T,A) to (N*self.encode_block_num, T//self.encode_block_num,A) \\
        after output, reshape action from (N*self.encode_block_num, T//self.encode_block_num,A) to (N,T,A) \\
        
        input: N,T,A
        output: N,T_latent,A
        """
        if isinstance(batch, dict):
            state = batch["action"]
        else:
            state = batch
        if reshape_in_vae:
            batch_size, T, action_dim = state.shape
            state = state.reshape(batch_size*self.encode_block_num, T//self.encode_block_num, action_dim)
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            raise NotImplementedError()
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
        state_vq = einops.rearrange(state_vq, 'N (T A) -> N T A', T=self.downsampled_input_h)
        if reshape_in_vae:
            state_vq = state_vq.reshape(batch_size, self.downsampled_input_h*self.encode_block_num, -1)
        return state_vq
    
    def decode_from_latent(self, action):
        """
        input: N,T_latent,A
        output: N,T,A
        """
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
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h_decode)
        dec_out = dec_out * self.act_scale
        dec_out = self.normalizer['action'].unnormalize(dec_out)

        return dec_out

            

    def encode_then_decode(self, batch):

        state = batch["action"]
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        batch_size, t_encode, action_dim = state.shape
        state = state.reshape(batch_size*self.encode_block_num,t_encode//self.encode_block_num, action_dim)
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            # Split latent policy output here
            state_vq = einops.rearrange(state_vq, "N (T A) -> N T A", T=self.downsampled_input_h).reshape(batch_size, self.downsampled_input_h*self.encode_block_num, -1).reshape(batch_size,-1)
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        # encoder_loss = (state - dec_out).abs().mean()
        dec_out = einops.rearrange(dec_out, "N (T A) -> N T A", T=self.input_dim_h_decode)
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
        self.encoder.train()
        self.decoder.train()
        if self.use_vq:
            self.vq_layer.train()
        else:
            self.quant.train()
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

    #### normalize latent for train latent-vla
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