from rtr_async_sys.models.reactive_diffusion_policy.model.vae.model import VAE
from rtr_async_sys.models.reactive_diffusion_policy.model.vae.utils import *
import einops
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

class Pi0_5_VAE(VAE):
    """
    normalize: match pi0.5 behavior. QUANTILES scales data to [-1, 1] using the 1st and 99th percentiles (q01/q99).
    1. normalize: use pi0.5 make_pre_post_processors to normalize and denormalize actions.
                  If make_pre_post_processors is unavailable, load the action statistics directly.
    2. latent_normalize: scan the dataset once to compute latent_statistics, normalize VAE latents as VLA inputs, then denormalize VLA outputs.
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

    def set_pre_post_processors(self, preprocessor, postprocessor):
        """
        pi0.5 pre/post processor normalizes and denormalizes actions. The VAE normalize/de_normalize methods operate on latents. \\
        batch = preprocessor(batch) \
        gt_actions = batch['action'];de_gt_actions = postprocessor(gt_actions) \
        """
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        

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
        if isinstance(batch, dict):
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
            # print(f"max is {torch.max(state_vq)} min is {torch.min(state_vq)}")
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
    

    def _load_dataset_statistics(self, path: Union[str, Path]=None, dataset_statistics:Dict = None) -> None:
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
        normalization_type has a large impact on accuracy.
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