import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl #import lightning.pytorch as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# Add casanovo and depthcharge to sys.path for imports
_project_root = Path(__file__).parent.parent  # MS-Spectral-Foundation/
_casanovo_path = _project_root / "casanovo"
_depthcharge_path = _project_root / "depthcharge"

if str(_casanovo_path) not in sys.path:
    sys.path.insert(0, str(_casanovo_path))
if str(_depthcharge_path) not in sys.path:
    sys.path.insert(0, str(_depthcharge_path))

# Import SpectrumEncoder from casanovo and FloatEncoder from depthcharge
from casanovo.denovo.transformers import SpectrumEncoder
from depthcharge.encoders import FloatEncoder

# Import bin_mz_tensor from Module 3
try:
    from MS_Spectral_Foundation.bin_mz import bin_mz_tensor
except ImportError:
    try:
        from bin_mz import bin_mz_tensor
    except ImportError:
        # Fallback: define a local version if import fails
        def bin_mz_tensor(mzs: torch.Tensor, *, bin_size: float, mz_min: float, mz_max: float, n_bins: int, is_normalized: bool = False) -> torch.LongTensor:
            """Local fallback version of bin_mz_tensor"""
            valid_mask = mzs > 0
            if not valid_mask.any():
                return torch.zeros_like(mzs, dtype=torch.long)
            if is_normalized:
                clamped = torch.clamp(mzs, min=0.0, max=1.0)
                bins = torch.floor(clamped * n_bins).long()
            else:
                clamped = torch.clamp(mzs, min=mz_min, max=mz_max)
                bins = torch.floor((clamped - mz_min) / bin_size).long()
            bins = bins.clamp(min=0, max=n_bins - 1)
            bins[~valid_mask] = 0
            return bins.detach()


class SpectrumSSLv2(pl.LightningModule):
    """
    Self-supervised spectrum modeling (encoder-only).
    Curriculum masking + binning + cross-entropy loss.
    Now masks m/z values instead of intensities and predicts m/z values.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-4,
        min_mask: float = 0.05,
        max_mask: float = 0.5,
        total_epochs: int = 100,
        mz_min: float = 50.0,
        mz_max: float = 2500.0,
        log_scale: bool = True,
        bin_size: float = 0.5,  

        max_peaks: int = 1000,
        # Masking config
        mask_prob: float = 0.15,  # used as overall probability; actual split is 80/10/10
        # Logging
        n_log: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        
        self.lr = lr
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.bin_size = bin_size
        self.MASK_TOKEN = -1.0 # # use -1.0 to distinguish from padding zeros

        ##  Data + masking
        self.max_peaks = max_peaks
        self.mask_prob = mask_prob
        # Masking schedule (from model_ssl_liu)
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.total_epochs = total_epochs


        # Auto-compute number of bins for DreaMS-style binning utility
        self.n_bins = int(math.ceil((self.mz_max - self.mz_min) / self.bin_size))
        print(f"[Init] n_bins automatically set to {self.n_bins}")

        # logging lists for plotting (per epoch)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Encoder
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Precursor encoders：Encodes charge、mass、mz
        # charge uses embeddings (because it's a small set of discrete values, typically 1-10)
        max_charge = 10  # Assume max charge is 10
        self.charge_encoder = nn.Embedding(max_charge + 1, dim_model)  # +1 to include 0-charge
        
        # mass and mz use FloatEncoder
        self.mass_encoder = FloatEncoder(d_model=dim_model)
        self.mz_encoder = FloatEncoder(d_model=dim_model)

        # MLP head for m/z regression (following model_ssl_v1.py structure)
        self.mz_predictor = nn.Sequential(
            nn.Linear(dim_model, dim_model // 4),  # uses dim_model // 4 to ensure it's an integer
            nn.ReLU(),
            nn.Linear(dim_model // 4, 1),  # Rectifies the dimension mismatch issue
        )
        self.mse_loss = nn.MSELoss()


        # Classification head – depends on n_bins
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 4),
            nn.ReLU(),
            nn.Linear(dim_model // 4, self.n_bins),
        )
        # Cross-Attention module: precursor attends to [CLS, peaks]
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=n_head,
            batch_first=True  # Make input and output in (B, L, D) format
        )
        self.cross_attn_ln = torch.nn.LayerNorm(dim_model)
        self.loss_fn = nn.CrossEntropyLoss()

        # Logging/plotting
        self.n_log = n_log
        self._history = []
        # Per-epoch aggregated losses for plotting
        self._plot_train_losses = []  # MSE (kept for backward compat)
        self._plot_val_losses = []    # MSE (kept for backward compat)
        self._plot_train_mse_losses = []
        self._plot_val_mse_losses = []
        self._plot_train_ce_losses = []
        self._plot_val_ce_losses = []
        self._plot_train_total_losses = []
        self._plot_val_total_losses = []

    # ------------------------------------------------------------------
    # Fixed-width binning (refer to DreaMS)
    # ------------------------------------------------------------------
    def bin_mz(self, mzs: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        """
        DreaMS-style fixed-width binning (left-closed, right-open intervals):
        - Uses floor for binning
        - Clamps m/z to [mz_min, mz_max] or [0, 1] if normalized
        - Bins limited to [0, n_bins-1]
        - Invalid (padded) positions (m/z <= 0) set to 0
        
        Args:
            mzs: m/z values tensor
            is_normalized: True if mzs are already normalized to [0,1], False if in original scale
        
        Returns:
            bins: torch.LongTensor, same shape as mzs
        """
        # Thin wrapper: delegate to bin_mz_tensor from Module 3
        return bin_mz_tensor(
            mzs,
            bin_size=self.bin_size,
            mz_min=self.mz_min,
            mz_max=self.mz_max,
            n_bins=self.n_bins,
            is_normalized=is_normalized
        )


    # ------------------------------------------------------------------
    # Masking strategy：Bert
    # ------------------------------------------------------------------
    def mask_spectrum(self, mzs: torch.Tensor, intensities: torch.Tensor, mask_rate: float):

        """
        Performs BERT-style masking on m/z values:
        - 80% -> Replace with MASK_TOKEN
        - 10% -> Replace with a random real peak from the same spectrum
        - 10% -> Keep original value
        """
        B, L = mzs.shape

        # Only mask valid peaks (m/z > 0)
        valid_mask = mzs > 0
        mask = torch.zeros_like(mzs, dtype=torch.bool)

        # Decide which positions to mask based on valid peaks
        for b in range(B):
            valid_indices = torch.where(valid_mask[b])[0]
            if len(valid_indices) > 0:
                n_mask = max(1, int(mask_rate * len(valid_indices)))
                chosen_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
                mask[b, chosen_indices] = True

        masked_mzs = mzs.clone()

        # Replacement strategy for masked positions with 80/10/10 split
        for b in range(B):
            masked_positions = torch.where(mask[b])[0]

            # Correct: recompute valid peaks for each sample to ensure random replacement is from the same spectrum
            valid_indices = torch.where(valid_mask[b])[0]
            valid_values = mzs[b, valid_indices]

            for pos in masked_positions:
                r = torch.rand(1).item()
                if r < 0.8:
                    masked_mzs[b, pos] = self.MASK_TOKEN
                elif r < 0.9:
                    # 10% → replace with a random real peak from the current spectrum
                    rand_idx = torch.randint(0, len(valid_values), (1,), device=mzs.device)
                    masked_mzs[b, pos] = valid_values[rand_idx]
                # elif r < 0.9:  # 10% probability: replace with a random value
                #     # generate a random value in [0, 1] for normalized m/z
                #     masked_mzs[b, pos] = torch.rand(1).item()  # sample uniformly in [0, 1]
                # # else: 10% probability: keep original value unchanged
                else:
                    # 10% → keep original value unchanged
                    pass

        return masked_mzs, mask

   

    # Returns full precursor info + pre-computed bin labels
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors (mzs, intensities, precursors, bin_labels).
        Returns full precursor info: [mass, charge, mz]
        Uses pre-computed bin labels (completed by Module 3 in Dataset)
        """
        # precursor_mzs = batch["precursor_mz"].squeeze(0)  # Shape: (batch_size,)
        # precursor_charges = batch["precursor_charge"].squeeze(0)  # Shape: (batch_size,)
        precursor_mzs = batch["precursor_mz"].float().view(-1)          # (B,)
        precursor_charges = batch["precursor_charge"].float().view(-1)  # (B,)
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges  # Shape: (batch_size,)
        
        # Combine into full precursor tensor: [mass, charge, mz]
        precursors = torch.stack([precursor_masses, precursor_charges, precursor_mzs], dim=1)  # Shape: (batch_size, 3)
        
        # Use pre-computed normalized m/z from Dataset (done in spectrum_dataset.py)
        normalized_mzs = batch["mz_normalized"]  # Shape: (batch_size, max_peaks), already [0,1]
        intensities = batch["intensity_array"]  # Shape: (batch_size, max_peaks)
        
        # Use pre-computed bin labels from Dataset (Module 3)
        bin_labels = batch["bin_labels"]  # Shape: (batch_size, max_peaks), long tensor
        
        # Apply the same normalization to precursor m/z
        normalized_precursors = precursors.clone()
        mz_range = self.mz_max - self.mz_min
        normalized_precursors[:, 2] = (precursors[:, 2] - self.mz_min) / mz_range  # normalize precursor mz
        
        return normalized_mzs, intensities, normalized_precursors, bin_labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, mzs: torch.Tensor, intensities: torch.Tensor, precursors: torch.Tensor):
        """
        Forward pass with full precursor information.
        Sequence layout: [CLS] + [PRECURSOR] + [PEAKS] = 1 + 1 + L tokens.
        
        Args:
            mzs: (B, L) m/z values for peaks
            intensities: (B, L) intensity values for peaks
            precursors: (B, 3) precursor info [mass, charge, mz]
            
        Returns:
            - predicted_mz: (B, L) continuous m/z predictions for all peaks
            - logits: (B, L, n_bins) classification logits for all peaks
            - padding_mask: (B, L+2) boolean mask for the full sequence (cls+precursor+peaks)
            - full_features: (B, L+2, d_model) embedding features for the full sequence
        """
        B, L = mzs.shape
        
        # 1. Encode each precursor component separately: mass, charge, mz
        # precursors: (B, 3) -> [mass, charge, mz]
        masses = precursors[:, 0]      # (B,) - mass values
        charges = precursors[:, 1]     # (B,) - charge values  
        mz_values = precursors[:, 2]   # (B,) - m/z values
        
        # Encode each component
        mass_encoded = self.mass_encoder(masses.unsqueeze(-1)).squeeze(1)  # (B, d_model)
        charge_encoded = self.charge_encoder(charges.long())  # (B, d_model) - cast charge to integer index
        mz_encoded = self.mz_encoder(mz_values.unsqueeze(-1)).squeeze(1)  # (B, d_model)
        
        # Sum the three encodings to get the final precursor representation
        precursors_combined = mass_encoded + charge_encoded + mz_encoded  # (B, d_model)
        precursor_token = precursors_combined.unsqueeze(1)  # (B, 1, d_model)
        
        # Part 1 - feature construction: encode [CLS+peaks] with self-attention into spectrum_memories,
        # then condition spectrum tokens on the precursor via cross-attention, and finally concatenate
        # into full_sequence = [CLS, precursor, peaks] for downstream prediction.
        # 2. Spectrum encoding (excluding precursor)
        spectrum_padding_mask = (mzs == 0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device)
        # Treat CLS as non-padding so it participates in self-attention but is not predicted
        spectrum_padding_mask = torch.cat([cls_mask, spectrum_padding_mask], dim=1) 
        spectrum_memories, _ = self.encoder(mzs, intensities, src_key_padding_mask=spectrum_padding_mask)  # (B, L, d_model)
        memory = spectrum_memories

        # === 2.5 Cross-Attention: Q=spectrum tokens, K/V=precursor token ===
        spectrum_q = memory               # (B, L+1, D)
        precursor_kv = precursor_token    # (B, 1, D)

        spectrum_refined, attn_weights = self.cross_attn(
            query=spectrum_q,
            key=precursor_kv,
            value=precursor_kv,
            key_padding_mask=None,   # precursor is a single token, no padding mask needed
        )
        memory = self.cross_attn_ln(memory + spectrum_refined)   # residual + LayerNorm for stability
        # 3. Concatenate sequence: [CLS] + [PRECURSOR] + [PEAKS...]
        # CLS is already handled internally by SpectrumEncoder;
        # insert the precursor token between CLS and the peak tokens.
        cls_token = memory[:, 0:1, :]  # (B, 1, d_model) - CLS token
        peak_tokens = memory[:, 1:, :]  # (B, L, d_model) - peak tokens
        
        # Concatenate: CLS + PRECURSOR + PEAKS
        full_sequence = torch.cat([cls_token, precursor_token, peak_tokens], dim=1)  # (B, L+2, d_model)
        
        # Part 2 - output + masking: attach a full padding mask to the concatenated sequence,
        # feed through prediction heads, then extract only the peaks portion for loss/evaluation.
        # 4. Build full padding mask: [CLS] + [PRECURSOR] + [PEAKS...]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device)       # CLS is not padding
        precursor_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device) # PRECURSOR is not padding
        peak_mask = spectrum_padding_mask[:, 1:]  # skip the CLS position
        full_padding_mask = torch.cat([cls_mask, precursor_mask, peak_mask], dim=1)  # (B, L+2)
        
        # 5. Prediction heads: use the full sequence (CLS + PRECURSOR + PEAKS)
        # so every peak can attend to CLS and precursor information for better prediction
        full_features = full_sequence  # (B, L+2, d_model)
        # Run decoder heads
        all_predicted_mz = self.mz_predictor(full_features).squeeze(-1)  # (B, L+2)
        all_logits = self.mlp_head(full_features)  # (B, L+2, n_bins)
        
        # Extract only the peaks portion for loss computation
        predicted_mz = all_predicted_mz[:, 2:]  # (B, L) - peaks only
        logits = all_logits[:, 2:, :]  # (B, L, n_bins) - peaks only
        
        return predicted_mz, logits, full_padding_mask, full_features

    # ------------------------------------------------------------------
    # Train/Val steps (regression)
    # ------------------------------------------------------------------

    # Training step with full precursor information
    def training_step(self, batch: Dict[str, torch.Tensor], *args, mode: str = "train") -> torch.Tensor:
        # Extract mzs, intensities, precursors, and pre-computed bin_labels
        mzs, intensities, precursors, labels = self._process_batch(batch)

        # Use fixed mask_prob as mask rate
        mask_rate = self.mask_prob

        # Every 100 steps, print m/z stats to verify normalization/range
        try:
            if self.global_step % 100 == 0:
                valid = mzs > 0
                if valid.any():
                    vals = mzs[valid]
                    vmin = vals.min().detach().item()
                    vmax = vals.max().detach().item()
                    vmean = vals.mean().detach().item()
                    sample = vals.flatten()[:10].detach().cpu().numpy()
                    print(f"[v4] step {self.global_step}: m/z min={vmin:.2f} max={vmax:.2f} mean={vmean:.2f} | sample={np.round(sample,2).tolist()}")
        except Exception:
            pass

        # Sort by m/z ascending (match model_ssl_liu.py behavior)
        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)
        labels = torch.gather(labels, 1, order)  # sort pre-computed labels in the same order

        # Labels are already pre-computed by Module 3; no need to call self.bin_mz() again
        
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)

        # Forward pass with full precursor information
        predicted_mz, logits, full_padding_mask, full_features = self.forward(masked_mzs, intensities, precursors)
        # predicted_mz: (B, L), logits: (B, L, n_bins), full_padding_mask: (B, L+2), full_features: (B, L+2, d_model)

        # Extract the peaks-only padding mask (skip CLS and PRECURSOR positions)
        padding_mask = full_padding_mask[:, 2:]  # (B, L)

        # Train on masked positions that are not padding
        train_mask = mask & (~padding_mask)
        # Early return if nothing to train on
        if train_mask.sum() == 0:
            zero = (predicted_mz.sum() + logits.sum()) * 0.0
            self.log(f"{mode}_Mask_Ratio", mask.float().mean().detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log(f"{mode}_mask_rate", torch.tensor(mask_rate, device=mzs.device), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            if mode == "train":
                self.log("train_regression_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
                self.log("train_classification_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
                self.log("train_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            return zero

        # Regression loss (MSE) on masked positions
        # Compute MSE directly in normalized space: m/z is already in [0,1],
        # so MSE is naturally in [0,1] range and comparable in magnitude to CE.
        raw_mse = self.mse_loss(predicted_mz[train_mask], mzs[train_mask])
        regression_loss = raw_mse  # use raw_mse directly; no additional scaling needed
        
        # RMSE in original m/z units for monitoring (rescale back to Th)
        mz_range = self.mz_max - self.mz_min
        rmse_mz_units = torch.sqrt(raw_mse) * mz_range  # convert back to Thomson units
        
        # Classification loss (CE) on masked positions
        masked_logits = logits[train_mask]  # (N_mask, n_bins)
        masked_labels = labels[train_mask]  # (N_mask,)
        # Safety check for label range
        assert masked_labels.max() < self.n_bins, \
            f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"

        classification_loss = self.loss_fn(masked_logits, masked_labels)
        
        # Dynamic weight schedule: classification-dominant early, regression-dominant late
        current_epoch = self.current_epoch
        total_epochs = self.total_epochs
        
        # Weight schedule: high CE weight early, high MSE weight late
        # epoch  0-30%: CE-dominant  (CE:1.0, MSE:0.3)
        # epoch 30-70%: linear transition (CE:1.0->0.5, MSE:0.3->1.0)
        # epoch 70-100%: MSE-dominant (CE:0.5, MSE:1.0)
        epoch_progress = current_epoch / total_epochs
        
        if epoch_progress < 0.3:
            # Early phase: classification-dominant
            ce_weight = 1.0
            mse_weight = 0.3
        elif epoch_progress < 0.7:
            # Middle phase: linear transition
            transition_progress = (epoch_progress - 0.3) / 0.4  # 0 to 1
            ce_weight = 1.0 - 0.5 * transition_progress  # 1.0 -> 0.5
            mse_weight = 0.3 + 0.7 * transition_progress  # 0.3 -> 1.0
        else:
            # Late phase: regression-dominant
            ce_weight = 0.5
            mse_weight = 1.0
            
        # Total loss = weighted combination
        weighted_regression_loss = mse_weight * regression_loss
        weighted_classification_loss = ce_weight * classification_loss
        total_loss = weighted_regression_loss + weighted_classification_loss

        # Log MSE and RMSE for interpretability
        self.log(f"{mode}_MSE_Loss", regression_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_CE_Loss", classification_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Total_Loss", total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # Log weights and weighted losses
        self.log(f"{mode}_CE_Weight", ce_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_MSE_Weight", mse_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Weighted_MSE_Loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Weighted_CE_Loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Raw_MSE", raw_mse.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # Log RMSE in original m/z units (rescaled)
        self.log(f"{mode}_RMSE_mz_units", rmse_mz_units.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        if mode == "train":
            self.log("train_regression_loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_classification_loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # Masking stats
        self.log(f"{mode}_Mask_Ratio", mask.float().mean().detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Log the effective mask_rate used this epoch
        self.log(f"{mode}_mask_rate", torch.tensor(mask_rate, device=mzs.device), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # return total_loss  # original: use weighted total loss
        return regression_loss  # back-propagate with MSE only

    # Validation step with full precursor information
    def validation_step(self, batch: Dict[str, torch.Tensor], *args) -> torch.Tensor:
        # Extract mzs, intensities, precursors, and pre-computed bin_labels
        mzs, intensities, precursors, labels = self._process_batch(batch)
        
        # Sort by m/z ascending to match training behavior
        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)
        labels = torch.gather(labels, 1, order)  # sort pre-computed labels in the same order
        
        # Use fixed mask_prob as mask rate
        mask_rate = self.mask_prob
        
        # Labels are already pre-computed by Module 3; no need to call self.bin_mz() again
        
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)
        
        # Forward pass with full precursor information
        predicted_mz, logits, full_padding_mask, full_features = self.forward(masked_mzs, intensities, precursors)

        # Extract the peaks-only padding mask (skip CLS and PRECURSOR positions).
        # mzs, labels, and mask retain their original shape to align with predicted_mz.
        padding_mask = full_padding_mask[:, 2:]  # (B, L)

        # Only masked and non-padding positions
        val_mask = mask & (~padding_mask)
        if val_mask.sum() == 0:
            zero = (predicted_mz.sum() + logits.sum()) * 0.0
            self.log("valid_MSE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("valid_CE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("val_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            return zero

        # Loss computation consistent with training_step
        raw_mse = self.mse_loss(predicted_mz[val_mask], mzs[val_mask])
        regression_loss = raw_mse  # use raw_mse directly; no additional scaling needed
        
        # RMSE in original m/z units for monitoring
        mz_range = self.mz_max - self.mz_min
        rmse_mz_units = torch.sqrt(raw_mse) * mz_range
        
        masked_logits = logits[val_mask]
        masked_labels = labels[val_mask]
        assert masked_labels.max() < self.n_bins, f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"
        classification_loss = self.loss_fn(masked_logits, masked_labels)
        
        # Same dynamic weight schedule as training_step
        current_epoch = self.current_epoch
        total_epochs = self.total_epochs
        epoch_progress = current_epoch / total_epochs
        
        if epoch_progress < 0.3:
            ce_weight = 1.0
            mse_weight = 0.3
        elif epoch_progress < 0.7:
            transition_progress = (epoch_progress - 0.3) / 0.4
            ce_weight = 1.0 - 0.5 * transition_progress
            mse_weight = 0.3 + 0.7 * transition_progress
        else:
            ce_weight = 0.5
            mse_weight = 1.0
            
        weighted_regression_loss = mse_weight * regression_loss
        weighted_classification_loss = ce_weight * classification_loss
        total_loss = weighted_regression_loss + weighted_classification_loss

        self.log("valid_MSE_Loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_CE_Loss", classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Log total loss explicitly for plotting
        self.log("valid_Total_Loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # Log weights and weighted losses
        self.log("valid_CE_Weight", ce_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_MSE_Weight", mse_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Weighted_MSE_Loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Weighted_CE_Loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Raw_MSE", raw_mse.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # Log RMSE in original m/z units (rescaled)
        self.log("valid_RMSE_mz_units", rmse_mz_units.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # self.log("val_loss", total_loss.detach(), ...)  # original: total loss
        self.log("val_loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # return total_loss  # original: use weighted total loss
        return regression_loss  # back-propagate with MSE only

    # ------------------------------------------------------------------
    # Epoch end hooks and plotting (adapted)
    # ------------------------------------------------------------------
    # Epoch-end hook: log metrics and save model on last epoch
    def on_train_epoch_end(self) -> None:
        callback_metrics = self.trainer.callback_metrics
        train_mse = callback_metrics.get("train_MSE_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        
        # Save model as .pt file on the final epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            pt_save_path = os.path.join(self.trainer.default_root_dir, f"model_ssl_v4_epoch{self.current_epoch:03d}.pt")
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.state_dict(),
                'hparams': self.hparams,
            }, pt_save_path)
            print(f"\n Model saved to {pt_save_path}")
        # Try to read CE/Total aggregated metrics if they exist
        try:
            train_ce = callback_metrics.get("train_CE_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_ce = float("nan")
        try:
            train_total = callback_metrics.get("train_Total_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_total = float("nan")

        # Validation epoch metrics (Lightning exposes them in callback_metrics)
        valid_mse = callback_metrics.get("valid_MSE_Loss", torch.tensor(float("nan"))).detach().item()
        try:
            valid_ce = callback_metrics.get("valid_CE_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_ce = float("nan")
        try:
            valid_total = callback_metrics.get("valid_Total_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_total = float("nan")

        mask_ratio = callback_metrics.get("train_Mask_Ratio", torch.tensor(float("nan"))).detach().item()

        # Unified epoch summary printing (train + valid, mse/ce/total)
        print(f"\nEpoch {self.current_epoch} Summary (SSL v4):")
        print(f"  Train MSE:   {train_mse:.6f}")
        print(f"  Train CE:    {train_ce:.6f}")
        print(f"  Train Total: {train_total:.6f}")
        print(f"  Valid MSE:   {valid_mse:.6f}")
        print(f"  Valid CE:    {valid_ce:.6f}")
        print(f"  Valid Total: {valid_total:.6f}")
        print(f"  Mask Ratio:  {mask_ratio:.2%}")

        metrics = {
            "step": self.trainer.global_step,
            "train_mse": train_mse,
            "mask_ratio": mask_ratio,
        }
        self._history.append(metrics)
        # Collect per-epoch aggregated metrics for plotting
        self._plot_train_losses.append(metrics.get("train_mse", float("nan")))  # legacy (MSE)
        self._plot_train_mse_losses.append(metrics.get("train_mse", float("nan")))
        self._plot_train_ce_losses.append(train_ce)
        # train Total loss if available
        try:
            train_total = self.trainer.callback_metrics.get("train_Total_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_total = float("nan")
        self._plot_train_total_losses.append(train_total)
        self._log_history()
        self._plot_loss_curves()

    def _plot_loss_curves(self) -> None:
        try:
            os.makedirs('logs', exist_ok=True)

            # MSE curves
            if len(self._plot_train_mse_losses) > 0 or len(self._plot_val_mse_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_mse_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_mse_losses) + 1), self._plot_train_mse_losses, 'b-', label='Train MSE', linewidth=2)
                if len(self._plot_val_mse_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_mse_losses) + 1), self._plot_val_mse_losses, 'r-', label='Valid MSE', linewidth=2)
                plt.title('SSL v4: MSE Loss')
                plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_mse.png'))
                plt.close()

            # CE curves
            if len(self._plot_train_ce_losses) > 0 or len(self._plot_val_ce_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_ce_losses) + 1), self._plot_train_ce_losses, 'b-', label='Train CE', linewidth=2)
                if len(self._plot_val_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_ce_losses) + 1), self._plot_val_ce_losses, 'r-', label='Valid CE', linewidth=2)
                plt.title('SSL v4: Cross-Entropy Loss')
                plt.xlabel('Epoch'); plt.ylabel('CE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_ce.png'))
                plt.close()

            # Total loss curves
            if len(self._plot_train_total_losses) > 0 or len(self._plot_val_total_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_total_losses) + 1), self._plot_train_total_losses, 'b-', label='Train Total', linewidth=2)
                if len(self._plot_val_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_total_losses) + 1), self._plot_val_total_losses, 'r-', label='Valid Total', linewidth=2)
                plt.title('SSL v4: Total Loss (MSE + CE)')
                plt.xlabel('Epoch'); plt.ylabel('Total Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_total.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_epochs,
                pct_start=0.3,  # warm-up phase covers 30% of total steps
                cycle_momentum=False
            ),
            'interval': 'epoch',  # update once per epoch
            'name': 'learning_rate'
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


    # Validation epoch-end hook: collect and log metrics
    def on_validation_epoch_end(self) -> None:
        callback_metrics = self.trainer.callback_metrics
        valid_mse = callback_metrics.get("valid_MSE_Loss", torch.tensor(float("nan"))).detach().item()
        # Collect CE if present
        try:
            valid_ce = callback_metrics.get("valid_CE_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_ce = float("nan")
        # Collect Total if present
        try:
            valid_total = callback_metrics.get("valid_Total_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_total = float("nan")
        metrics = {"step": self.trainer.global_step, "valid_mse": valid_mse}
        self._history.append(metrics)
        self._plot_val_losses.append(metrics.get("valid_mse", float("nan")))  # legacy (MSE)
        self._plot_val_mse_losses.append(metrics.get("valid_mse", float("nan")))
        self._plot_val_ce_losses.append(valid_ce)
        self._plot_val_total_losses.append(valid_total)
        self._log_history()
        self._plot_loss_curves()

    # Extended history logging for debugging
    def _log_history(self) -> None:
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            print("Step\tTrain MSE\tValid MSE\tMask%")
        metrics = self._history[-1]
        if metrics.get("step", 0) % self.n_log == 0:
            vals = [
                metrics.get("step", -1),
                metrics.get("train_mse", float("nan")),
                metrics.get("valid_mse", float("nan")),
                (metrics.get("mask_ratio", float("nan")) or 0.0) * 100,
            ]
            print("%i\t%.6f\t%.6f\t%.2f" % tuple(vals))
    
    @torch.no_grad()
    def get_embeddings(self, batch, precursor_only=False):
        self.eval()
        mzs, intensities, precursors, _ = self._process_batch(batch)  # bin_labels not needed here
        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)
        _, _, _, full_features = self.forward(mzs, intensities, precursors)

        CLS = full_features[:, 0, :]
        PREC = full_features[:, 1, :]
        PEAKS = full_features[:, 2:, :]

        if precursor_only:
            return PREC
        return CLS

# Backward-compatible alias if desired
SpectrumSSL = SpectrumSSLv2