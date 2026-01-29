"""
Bitwav training pipeline using PyTorch Lightning.
Handles multi-stage training including feature reconstruction and GAN-based mel generation.
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List, Dict, Any

import jsonargparse
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .data.datamodule import AudioBatch
from .model import BitwavModel, BitwavModelConfig
from .module.audio_feature import MelSpectrogramFeature
from .module.discriminator import SpectrogramDiscriminator
from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .util import freeze_modules, get_logger, load_vocoder, vocode

logger = get_logger()


@dataclass
class BitwavPipelineConfig:
    """
    Configuration for the Bitwav training pipeline.

    Attributes:
        train_feature (bool): Whether to train the SSL feature reconstruction branch.
        train_mel (bool): Whether to train the mel spectrogram generation branch.
        audio_length (int): Length of input audio in samples for training.
        lr (float): Peak learning rate for the generator.
        weight_decay (float): Weight decay for optimization.
        betas (Tuple[float, float]): Beta parameters for AdamW.
        gradient_clip_val (float | None): Gradient clipping value.
        warmup_percent (float): Percentage of total steps for LR warmup.
        lr_div_factor (float): Initial division factor for OneCycleLR.
        lr_final_div_factor (float): Final division factor for OneCycleLR.
        anneal_mode (str): Annealing strategy for OneCycleLR ('cos' or 'linear').
        feature_l1_weight (float): L1 loss weight for feature reconstruction.
        feature_l2_weight (float): L2 loss weight for feature reconstruction.
        mel_l1_weight (float): L1 loss weight for mel reconstruction.
        mel_l2_weight (float): L2 loss weight for mel reconstruction.
        adv_weight (float): Weight for adversarial loss.
        feature_matching_weight (float): Weight for feature matching loss.
        use_discriminator (bool): Whether to enable GAN training.
        adv_loss_type (Literal["hinge", "least_square"]): Type of GAN loss.
        discriminator_lr (float | None): Learning rate for the discriminator.
        discriminator_start_step (int): Training step at which to start discriminator updates.
        discriminator_update_prob (float): Probability of updating the discriminator in a given step.
        ckpt_path (str | None): Path to a checkpoint to load weights from.
        skip_loading_modules (Tuple[str, ...]): Modules to skip when loading from a checkpoint.
        log_mel_samples (int): Number of mel spectrogram samples to log during validation.
        use_torch_compile (bool): Whether to use torch.compile for performance.
    """
    train_feature: bool = True
    train_mel: bool = True

    audio_length: int = 138240

    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    gradient_clip_val: Optional[float] = 1.0

    warmup_percent: float = 0.1
    lr_div_factor: float = 10.0
    lr_final_div_factor: float = 1.0
    anneal_mode: str = "cos"

    feature_l1_weight: float = 30.0
    feature_l2_weight: float = 0.0
    mel_l1_weight: float = 30.0
    mel_l2_weight: float = 0.0
    adv_weight: float = 1.0
    feature_matching_weight: float = 10.0

    use_discriminator: bool = False
    adv_loss_type: Literal["hinge", "least_square"] = "hinge"
    discriminator_lr: Optional[float] = None
    discriminator_start_step: int = 0
    discriminator_update_prob: float = 1.0

    ckpt_path: Optional[str] = None
    skip_loading_modules: Tuple[str, ...] = ()

    log_mel_samples: int = 10
    use_torch_compile: bool = True


class BitwavPipeline(L.LightningModule):
    """
    LightningModule for training and validating Bitwav models.
    Supports adversarial training with a spectrogram discriminator.
    """

    def __init__(
        self,
        model_config: BitwavModelConfig,
        pipeline_config: BitwavPipelineConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Optional[Transformer],
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer,
        mel_decoder: Transformer,
        mel_postnet: PostNet,
        discriminator: Optional[SpectrogramDiscriminator] = None,
    ):
        """
        Initializes the BitwavPipeline.
        """
        super().__init__()
        self.config = pipeline_config
        self.save_hyperparameters("model_config", "pipeline_config")
        self.strict_loading = False
        self.automatic_optimization = False
        self.torch_compiled = False

        if pipeline_config.train_feature:
            assert feature_decoder is not None, "Feature decoder required for feature training."

        self.model = BitwavModel(
            config=model_config,
            ssl_feature_extractor=ssl_feature_extractor,
            local_encoder=local_encoder,
            local_quantizer=local_quantizer,
            feature_decoder=feature_decoder,
            global_encoder=global_encoder,
            mel_decoder=mel_decoder,
            mel_prenet=mel_prenet,
            mel_postnet=mel_postnet,
        )
        self._freeze_unused_modules(pipeline_config.train_feature, pipeline_config.train_mel)

        self.padding = self.model._calculate_waveform_padding(pipeline_config.audio_length)
        self.target_mel_length = self.model._calculate_target_mel_length(pipeline_config.audio_length)

        self._init_discriminator(pipeline_config, discriminator)

        if pipeline_config.train_mel:
            self.mel_spec = MelSpectrogramFeature(
                sample_rate=model_config.sample_rate,
                n_fft=model_config.n_fft,
                hop_length=model_config.hop_length,
                n_mels=model_config.n_mels,
                padding=model_config.padding,
                fmin=model_config.mel_fmin,
                fmax=model_config.mel_fmax,
                bigvgan_style_mel=model_config.bigvgan_style_mel,
            )

        self.vocoder = None
        self.validation_examples = []
        self.log_mel_samples = pipeline_config.log_mel_samples

    def _freeze_unused_modules(self, train_feature: bool, train_mel: bool):
        """Freezes model branches based on training flags."""
        model = self.model
        if not train_feature:
            freeze_modules([model.local_encoder, model.local_quantizer, model.feature_decoder])
            if model.conv_downsample is not None:
                freeze_modules([model.conv_downsample, model.conv_upsample])
        if not train_mel:
            freeze_modules([model.global_encoder, model.mel_prenet, model.mel_conv_upsample, model.mel_decoder, model.mel_postnet])

    def _init_discriminator(self, config: BitwavPipelineConfig, discriminator: Optional[SpectrogramDiscriminator]):
        """Configures the discriminator for GAN training."""
        self.discriminator = discriminator
        self.use_discriminator = config.use_discriminator and discriminator is not None and config.train_mel
        self.discriminator_start_step = config.discriminator_start_step
        self.discriminator_update_prob = config.discriminator_update_prob

    def setup(self, stage: str):
        """Final setup: optional torch compilation and weight loading."""
        if torch.__version__ >= "2.0" and self.config.use_torch_compile:
            self.model = torch.compile(self.model)
            if self.discriminator is not None:
                self.discriminator = torch.compile(self.discriminator)
            self.torch_compiled = True

        if self.config.ckpt_path:
            ckpt_path = self.config.ckpt_path
            if ckpt_path.startswith("hf:"):
                from huggingface_hub import hf_hub_download
                repo_id = ckpt_path[len("hf:") :]
                revision = None
                if "@" in repo_id:
                    repo_id, revision = repo_id.split("@", 1)
                ckpt_path = hf_hub_download(repo_id, filename="model.safetensors", revision=revision)
            self._load_weights(ckpt_path)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Core forward pass used in training steps."""
        loss_dict = {}
        local_ssl, global_ssl = self.model.forward_ssl_features(waveform, padding=self.padding)
        content_embeddings, _, ssl_recon, perplexity = self.model.forward_content(local_ssl)
        loss_dict["local/perplexity"] = perplexity

        mel_recon = None
        if self.config.train_mel:
            global_embeddings = self.model.forward_global(global_ssl)
            mel_recon = self.model.forward_mel(content_embeddings, global_embeddings, mel_length=self.target_mel_length)

        return local_ssl, ssl_recon, mel_recon, loss_dict

    def _get_reconstruction_loss(
        self, audio_real: torch.Tensor, ssl_real: torch.Tensor, ssl_recon: Optional[torch.Tensor], mel_recon: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Calculates reconstruction loss for SSL and mel spectrograms."""
        if audio_real.dim() == 3:
            audio_real = audio_real.squeeze(1)

        loss_dict = {}
        feature_loss, mel_loss = 0, 0

        if self.config.train_feature and ssl_recon is not None:
            l1 = F.l1_loss(ssl_recon, ssl_real)
            l2 = F.mse_loss(ssl_recon, ssl_real)
            feature_loss = self.config.feature_l1_weight * l1 + self.config.feature_l2_weight * l2
            loss_dict.update({"ssl_l1": l1, "ssl_l2": l2, "feature_loss": feature_loss})

        mel_real = None
        if self.config.train_mel and mel_recon is not None:
            mel_real = self.mel_spec(audio_real)
            l1 = F.l1_loss(mel_recon, mel_real)
            l2 = F.mse_loss(mel_recon, mel_real)
            mel_loss = self.config.mel_l1_weight * l1 + self.config.mel_l2_weight * l2
            loss_dict.update({"mel_l1": l1, "mel_l2": l2, "mel_loss": mel_loss})

        return feature_loss + mel_loss, loss_dict, mel_real

    def _get_discriminator_loss(self, real_outputs: torch.Tensor, fake_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adversarial loss for discriminator."""
        if self.config.adv_loss_type == "hinge":
            real_loss = torch.mean(torch.clamp(1 - real_outputs, min=0))
            fake_loss = torch.mean(torch.clamp(1 + fake_outputs, min=0))
        else: # least_square
            real_loss = torch.mean((real_outputs - 1) ** 2)
            fake_loss = torch.mean(fake_outputs**2)
        return real_loss + fake_loss, real_loss, fake_loss

    def _get_generator_loss(self, fake_outputs: torch.Tensor) -> torch.Tensor:
        """Adversarial loss for generator."""
        if self.config.adv_loss_type == "hinge":
            return torch.mean(torch.clamp(1 - fake_outputs, min=0))
        return torch.mean((fake_outputs - 1) ** 2)

    def _get_feature_matching_loss(self, real_ints: List[torch.Tensor], fake_ints: List[torch.Tensor]) -> torch.Tensor:
        """Feature matching loss between real and fake hidden discriminator states."""
        losses = [torch.mean(torch.abs(r.detach() - f)) for r, f in zip(real_ints, fake_ints)]
        return torch.mean(torch.stack(losses))

    def _discriminator_step(self, batch: AudioBatch, optimizer_disc: torch.optim.Optimizer) -> Tuple:
        """Single optimization step for the discriminator."""
        ssl_real, ssl_recon, mel_recon, loss_dict = self(batch.waveform)
        mel_real = self.mel_spec(batch.waveform)

        real_out, real_ints = self.discriminator(mel_real)
        fake_out, _ = self.discriminator(mel_recon.detach())
        disc_loss, real_loss, fake_loss = self._get_discriminator_loss(real_out, fake_out)

        bsz = batch.waveform.size(0)
        self.log("train/disc/loss", disc_loss, batch_size=bsz, prog_bar=True)
        
        optimizer_disc.zero_grad()
        self.manual_backward(disc_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip_val or torch.inf)
        optimizer_disc.step()

        return ssl_real, ssl_recon, mel_recon, loss_dict, real_ints

    def _generator_step(self, batch: AudioBatch, optimizer_gen: torch.optim.Optimizer, 
                        ssl_real=None, ssl_recon=None, mel_recon=None, loss_dict=None, real_ints=None, training_disc=False):
        """Single optimization step for the generator."""
        if loss_dict is None:
            ssl_real, ssl_recon, mel_recon, loss_dict = self(batch.waveform)

        recon_loss, recon_dict, mel_real = self._get_reconstruction_loss(batch.waveform, ssl_real, ssl_recon, mel_recon)
        gen_loss = recon_loss

        bsz = batch.waveform.size(0)
        if training_disc:
            if real_ints is None:
                _, real_ints = self.discriminator(mel_real)
            fake_out, fake_ints = self.discriminator(mel_recon)
            
            adv_loss = self._get_generator_loss(fake_out)
            fm_loss = self._get_feature_matching_loss(real_ints, fake_ints)
            gen_loss += self.config.adv_weight * adv_loss + self.config.feature_matching_weight * fm_loss
            self.log("train/gen/adv_loss", adv_loss, batch_size=bsz)

        self.log("train/loss", gen_loss, batch_size=bsz, prog_bar=True)
        
        optimizer_gen.zero_grad()
        self.manual_backward(gen_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val or torch.inf)
        optimizer_gen.step()

        return gen_loss

    def training_step(self, batch: AudioBatch, batch_idx: int):
        """Main training step coordinating GAN updates."""
        optims = self.optimizers()
        schs = self.lr_schedulers()
        
        if self.use_discriminator:
            optimizer_disc, optimizer_gen = optims
            scheduler_disc, scheduler_gen = schs
        else:
            optimizer_gen = optims
            scheduler_gen = schs

        training_disc = (self.use_discriminator and self.global_step >= self.discriminator_start_step and 
                        torch.rand(1).item() < self.discriminator_update_prob)

        ssl_real, ssl_recon, mel_recon, loss_dict, real_ints = None, None, None, None, None

        if training_disc:
            ssl_real, ssl_recon, mel_recon, loss_dict, real_ints = self._discriminator_step(batch, optimizer_disc)
            scheduler_disc.step()
        elif self.use_discriminator:
            scheduler_disc.step()

        self._generator_step(batch, optimizer_gen, ssl_real, ssl_recon, mel_recon, loss_dict, real_ints, training_disc)
        scheduler_gen.step()

    def validation_step(self, batch: AudioBatch, batch_idx: int):
        """Validates the model and prepares spectrograms for logging."""
        ssl_real, ssl_recon, mel_recon, loss_dict = self(batch.waveform)
        recon_loss, recon_dict, mel_real = self._get_reconstruction_loss(batch.waveform, ssl_real, ssl_recon, mel_recon)
        
        bsz = batch.waveform.size(0)
        self.log("val/loss", recon_loss, batch_size=bsz)

        if self.config.train_mel and len(self.validation_examples) < self.log_mel_samples:
            example_audio_real = batch.waveform[0].cpu()
            example_audio_gen = None
            if self.vocoder is not None:
                with torch.no_grad():
                    example_audio_gen = self.vocode(mel_recon[0:1])[0].cpu()
            self.validation_examples.append((mel_real[0].cpu(), mel_recon[0].detach().cpu(), example_audio_real, example_audio_gen))

    def predict_step(self, batch: AudioBatch, batch_idx: int):
        """Performs inference (waveform generation)."""
        _, _, mel_gen, _ = self(batch.waveform)
        audio_gen = self.vocode(mel_gen)
        if audio_gen.dim() == 2:
            audio_gen = audio_gen.unsqueeze(1)
        return {"audio_ids": batch.audio_ids, "audio_real": batch.waveform, "audio_gen": audio_gen}

    def configure_optimizers(self):
        """Configures optimizers and OneCycleLR schedulers."""
        opt_gen = AdamW(self.model.parameters(), lr=self.config.lr, betas=self.config.betas, weight_decay=self.config.weight_decay)
        sch_gen = OneCycleLR(opt_gen, max_lr=self.config.lr, total_steps=self.trainer.estimated_stepping_batches,
                             pct_start=self.config.warmup_percent, anneal_strategy=self.config.anneal_mode,
                             div_factor=self.config.lr_div_factor, final_div_factor=self.config.lr_final_div_factor)

        if not self.use_discriminator:
            return [opt_gen], [{"scheduler": sch_gen, "interval": "step"}]

        opt_disc = AdamW(self.discriminator.parameters(), lr=self.config.discriminator_lr or self.config.lr, 
                          betas=self.config.betas, weight_decay=self.config.weight_decay)
        sch_disc = OneCycleLR(opt_disc, max_lr=self.config.discriminator_lr or self.config.lr, 
                              total_steps=self.trainer.estimated_stepping_batches,
                              pct_start=self.config.warmup_percent, anneal_strategy=self.config.anneal_mode,
                              div_factor=self.config.lr_div_factor, final_div_factor=self.config.lr_final_div_factor)

        return [opt_disc, opt_gen], [{"scheduler": sch_disc, "interval": "step"}, {"scheduler": sch_gen, "interval": "step"}]

    def _setup_vocoder(self):
        """Loads the vocoder for audio logging."""
        try:
            return load_vocoder(name=self.model.config.vocoder_name)
        except Exception:
            return None

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocodes mel spectrograms to waveforms."""
        self.vocoder = self.vocoder.to(mel.device)
        return vocode(self.vocoder, mel).cpu().float()

    def on_validation_start(self):
        self.vocoder = self._setup_vocoder()

    def on_predict_start(self):
        self.vocoder = self._setup_vocoder()

    def on_validation_end(self):
        """Logs validation spectrograms and audio to the logger (TB/WandB)."""
        if self.validation_examples:
            for i, (m_real, m_gen, a_real, a_gen) in enumerate(self.validation_examples):
                f_real = self._get_spectrogram_plot(m_real)
                f_gen = self._get_spectrogram_plot(m_gen)
                self._log_figure(f"val/{i}_mel_real", f_real)
                self._log_figure(f"val/{i}_mel_gen", f_gen)
                if a_gen is not None:
                    self._log_audio(f"val/{i}_audio_real", a_real.numpy())
                    self._log_audio(f"val/{i}_audio_gen", a_gen.numpy())
            self.validation_examples = []
        self.vocoder = None

    def _log_figure(self, tag, fig):
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(tag, fig, self.global_step)
        elif isinstance(self.logger, WandbLogger):
            import PIL.Image as Image
            fig.canvas.draw()
            image = Image.frombytes("RGBa", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            self.logger.log_image(tag, [image], step=self.global_step)

    def _log_audio(self, tag, audio):
        sr = self.model.config.sample_rate
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_audio(tag, audio, self.global_step, sample_rate=sr)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_audio(tag, [audio.flatten()], sample_rate=[sr], step=self.global_step)

    def _get_spectrogram_plot(self, mel):
        from matplotlib import pyplot as plt
        mel = mel.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", vmin=-8.0, vmax=5.0)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def _load_weights(self, ckpt_path: str):
        """Loads weights from .ckpt, .safetensors, or .pt files."""
        if ckpt_path.endswith(".ckpt"):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            m_sd = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
            d_sd = {k[len("discriminator."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("discriminator.")}
        elif ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            m_sd = load_file(ckpt_path, device="cpu")
            d_sd = {}
        else:
            m_sd = torch.load(ckpt_path, map_location="cpu")
            d_sd = {}

        m_sd = {k: v for k, v in m_sd.items() if not any(k.startswith(m) for m in self.config.skip_loading_modules)}
        self.model.load_state_dict(m_sd, strict=False)
        if d_sd and self.use_discriminator:
            self.discriminator.load_state_dict(d_sd, strict=False)

    @classmethod
    def from_hparams(cls, config_path: str) -> "BitwavPipeline":
        """Instantiates BitwavPipeline from a YAML config."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        new_config = {"model": config["model"]}
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=BitwavPipeline)
        cfg = parser.parse_object(new_config)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @staticmethod
    def from_pretrained(config_path: str, ckpt_path: str) -> "BitwavPipeline":
        """Instantiates and loads weights for a BitwavPipeline."""
        model = BitwavPipeline.from_hparams(config_path)
        model._load_weights(ckpt_path)
        return model
