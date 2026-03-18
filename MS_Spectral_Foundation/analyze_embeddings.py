"""
Module 7 & 8: Embedding Analysis and Downstream Evaluation

Full data flow:
Module 1-3 (preprocessing) → Module 4-6 (SSL training) → Module 7-8 (embedding analysis)

This module:
- Extracts spectrum-level embeddings from pretrained model (Module 6)
- Performs sample-level aggregation (Module 7.1)
- Analyzes embedding-space distributions (Module 7.2)
- Discovers exemplar spectra (Module 7.3)
- Evaluates representations without supervision (Module 8)
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("  UMAP not available. Install with: pip install umap-learn")

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from MS_Spectral_Foundation.spectrum_dataset import SpectrumDataset
from MS_Spectral_Foundation.model_ssl import SpectrumSSLv2


class EmbeddingAnalyzer:
    """
    Analyze learned spectrum embeddings for downstream tasks.
    
    Implements:
    - Module 7: Embedding-space distribution analysis and sample-level structure
    - Module 8: Non-supervised representation evaluation
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize analyzer with pretrained model.
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"   Model loaded from {model_path}")
        print(f"   Device: {self.device}")
        
    def _load_model(self, model_path: str) -> SpectrumSSLv2:
        """Load pretrained model from checkpoint."""
        path = Path(model_path)
        
        if path.suffix == ".pt":
            # Custom .pt checkpoint
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            hparams = checkpoint.get("hparams", {})
            
            model = SpectrumSSLv2(**hparams)
            model.load_state_dict(checkpoint["model_state_dict"])
            
        elif path.suffix == ".ckpt":
            # PyTorch Lightning checkpoint
            model = SpectrumSSLv2.load_from_checkpoint(
                model_path,
                map_location="cpu"
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {path.suffix}")
        
        return model
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataset: SpectrumDataset,
        batch_size: int = 64,
        num_workers: int = 4,
        embedding_type: str = "cls"
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectrum-level embeddings (Module 6).
        
        Args:
            dataset: SpectrumDataset to extract embeddings from
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            embedding_type: Type of embedding to extract
                - "cls": CLS token embedding (default)
                - "precursor": Precursor token embedding
                - "mean_peaks": Mean pooling over peak tokens
        
        Returns:
            Dictionary containing:
            - embeddings: (N, d) array of spectrum embeddings
            - precursor_mz: (N,) array of precursor m/z values
            - precursor_charge: (N,) array of precursor charges
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        embeddings_list = []
        precursor_mz_list = []
        precursor_charge_list = []
        
        print(f"\n[Module 6] Extracting {embedding_type} embeddings...")
        print(f"  Total spectra: {len(dataset)}")
        
        for batch_idx, batch in enumerate(loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Extract embeddings using model's get_embeddings method
            if embedding_type == "precursor":
                emb = self.model.get_embeddings(batch, precursor_only=True)
            elif embedding_type == "cls":
                emb = self.model.get_embeddings(batch, precursor_only=False)
            elif embedding_type == "mean_peaks":
                # Extract full features and mean pool over peaks
                mzs, intensities, precursors, _ = self.model._process_batch(batch)
                _, _, _, full_features = self.model.forward(mzs, intensities, precursors)
                peak_features = full_features[:, 2:, :]  # Skip CLS and precursor
                # Mask out padded peaks
                mask = (mzs > 0).unsqueeze(-1)  # (B, L, 1)
                peak_features = peak_features * mask
                emb = peak_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown embedding_type: {embedding_type}")
            
            embeddings_list.append(emb.cpu().numpy())
            precursor_mz_list.append(batch["precursor_mz"].cpu().numpy().squeeze())
            precursor_charge_list.append(batch["precursor_charge"].cpu().numpy().squeeze())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} spectra")
        
        embeddings = np.vstack(embeddings_list)
        precursor_mz = np.concatenate(precursor_mz_list)
        precursor_charge = np.concatenate(precursor_charge_list)
        
        print(f" Extracted embeddings: {embeddings.shape}")
        
        return {
            "embeddings": embeddings,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
        }
    
    def aggregate_to_sample_level(
        self,
        embeddings: np.ndarray,
        sample_ids: np.ndarray,
        method: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate spectrum embeddings to sample-level (Module 7.1).
        
        Args:
            embeddings: (N, d) spectrum embeddings
            sample_ids: (N,) array of sample identifiers
            method: Aggregation method ("mean" or "median")
        
        Returns:
            - sample_embeddings: (n_samples, d)
            - unique_sample_ids: (n_samples,)
        """
        print(f"\n[Module 7.1] Sample-level aggregation (method: {method})")
        
        unique_samples = np.unique(sample_ids)
        n_samples = len(unique_samples)
        d = embeddings.shape[1]
        
        sample_embeddings = np.zeros((n_samples, d))
        
        for i, sample_id in enumerate(unique_samples):
            mask = sample_ids == sample_id
            sample_spectra = embeddings[mask]
            
            if method == "mean":
                sample_embeddings[i] = sample_spectra.mean(axis=0)
            elif method == "median":
                sample_embeddings[i] = np.median(sample_spectra, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            if (i + 1) % 10 == 0 or (i + 1) == n_samples:
                print(f"  Aggregated {i + 1}/{n_samples} samples")
        
        print(f"   Sample-level embeddings: {sample_embeddings.shape}")
        print(f"   Spectra per sample: mean={len(sample_ids)/n_samples:.1f}")
        
        return sample_embeddings, unique_samples
    
    def analyze_embedding_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: Dict[int, str] = None,
        max_samples: Optional[int] = 2000
    ) -> Dict[str, float]:
        """
        Analyze embedding distributions between groups (Module 7.2).
        
        Computes:
        - Intra-group distances (within HCC, within Cirrhosis)
        - Inter-group distances (HCC vs Cirrhosis)
        - Separation metrics
        
        Args:
            embeddings: (N, d) embeddings
            labels: (N,) group labels (0, 1, ...)
            label_names: Optional mapping {0: "HCC", 1: "Cirrhosis", ...}
            max_samples: If N > max_samples, subsample each group before computing
                distances to avoid OOM. Default 2000.
        
        Returns:
            Dictionary of distance metrics
        """
        print(f"\n[Module 7.2] Embedding distribution analysis")
        
        unique_labels = np.unique(labels)
        n_groups = len(unique_labels)

        # Subsample each group to avoid OOM on large datasets
        if max_samples is not None and len(embeddings) > max_samples * n_groups:
            rng = np.random.default_rng(42)
            keep = []
            for lbl in unique_labels:
                idx = np.where(labels == lbl)[0]
                chosen = rng.choice(idx, size=min(max_samples, len(idx)), replace=False)
                keep.append(chosen)
            keep = np.sort(np.concatenate(keep))
            embeddings = embeddings[keep]
            labels = labels[keep]
            print(f"  (Subsampled to {len(embeddings)} spectra for distance computation)")
        
        if label_names is None:
            label_names = {i: f"Group_{i}" for i in unique_labels}
        
        metrics = {}
        
        # Compute pairwise distances
        dist_matrix = squareform(pdist(embeddings, metric="euclidean"))
        
        # Intra-group distances
        for label in unique_labels:
            mask = labels == label
            intra_dists = dist_matrix[np.ix_(mask, mask)]
            # Remove diagonal (self-distances)
            intra_dists = intra_dists[np.triu_indices_from(intra_dists, k=1)]
            
            name = label_names[label]
            metrics[f"intra_{name}_mean"] = intra_dists.mean()
            metrics[f"intra_{name}_std"] = intra_dists.std()
            
            print(f"  Intra-group ({name}): {intra_dists.mean():.4f} ± {intra_dists.std():.4f}")
        
        # Inter-group distances
        if n_groups == 2:
            label0, label1 = unique_labels
            mask0 = labels == label0
            mask1 = labels == label1
            
            inter_dists = dist_matrix[np.ix_(mask0, mask1)].flatten()
            
            name0 = label_names[label0]
            name1 = label_names[label1]
            
            metrics[f"inter_{name0}_{name1}_mean"] = inter_dists.mean()
            metrics[f"inter_{name0}_{name1}_std"] = inter_dists.std()
            
            print(f"  Inter-group ({name0} vs {name1}): {inter_dists.mean():.4f} ± {inter_dists.std():.4f}")
            
            # Separation ratio: inter / mean(intra)
            mean_intra = (metrics[f"intra_{name0}_mean"] + metrics[f"intra_{name1}_mean"]) / 2
            separation_ratio = metrics[f"inter_{name0}_{name1}_mean"] / mean_intra
            metrics["separation_ratio"] = separation_ratio
            
            print(f"  Separation ratio (inter/intra): {separation_ratio:.4f}")
            
            # Statistical test (Mann-Whitney U)
            # Compare distances: intra-group vs inter-group
            intra0 = dist_matrix[np.ix_(mask0, mask0)][np.triu_indices(mask0.sum(), k=1)]
            intra1 = dist_matrix[np.ix_(mask1, mask1)][np.triu_indices(mask1.sum(), k=1)]
            all_intra = np.concatenate([intra0, intra1])
            
            if len(all_intra) > 0 and len(inter_dists) > 0:
                stat, pval = mannwhitneyu(all_intra, inter_dists, alternative='two-sided')
                metrics["mannwhitney_u_statistic"] = stat
                metrics["mannwhitney_p_value"] = pval
                print(f"  Mann-Whitney U test (intra vs inter): p={pval:.4e}")
            else:
                print(f"  Mann-Whitney U test: skipped (insufficient intra-group samples)")
        
        print(f" Distribution analysis complete")
        
        return metrics
    
    def find_exemplar_spectra(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        dataset: Optional[SpectrumDataset],
        n_exemplars: int = 5,
        strategy: str = "extremes"
    ) -> Dict[str, List[int]]:
        """
        Discover representative exemplar spectra (Module 7.3).
        
        Args:
            embeddings: (N, d) spectrum embeddings
            labels: (N,) group labels
            dataset: Original dataset (to retrieve spectra)
            n_exemplars: Number of exemplar spectra per category
            strategy: Selection strategy
                - "extremes": Spectra with max/min cross-group distance
                - "centroids": Spectra closest to group centroids
                - "boundary": Spectra near decision boundary
        
        Returns:
            Dictionary mapping categories to spectrum indices
        """
        print(f"\n[Module 7.3] Exemplar spectra discovery (strategy: {strategy})")
        
        unique_labels = np.unique(labels)
        exemplars = {}
        
        if strategy == "centroids":
            # Find spectra closest to group centroids
            for label in unique_labels:
                mask = labels == label
                group_emb = embeddings[mask]
                centroid = group_emb.mean(axis=0)
                
                # Compute distances to centroid
                dists = np.linalg.norm(group_emb - centroid, axis=1)
                closest_indices = np.argsort(dists)[:n_exemplars]
                
                # Map back to original dataset indices
                original_indices = np.where(mask)[0][closest_indices]
                exemplars[f"group_{label}_centroid"] = original_indices.tolist()
                
                print(f"  Group {label} centroids: {len(original_indices)} spectra")
        
        elif strategy == "extremes" and len(unique_labels) == 2:
            # Max and min cross-group distance
            label0, label1 = unique_labels
            mask0 = labels == label0
            mask1 = labels == label1
            
            emb0 = embeddings[mask0]
            emb1 = embeddings[mask1]
            
            # Subsample to avoid OOM (full matrix can be hundreds of GiB)
            max_per_group = 2000
            rng = np.random.default_rng(42)
            idx0 = rng.choice(len(emb0), size=min(max_per_group, len(emb0)), replace=False)
            idx1 = rng.choice(len(emb1), size=min(max_per_group, len(emb1)), replace=False)
            emb0_sub = emb0[idx0]
            emb1_sub = emb1[idx1]
            orig_idx0 = np.where(mask0)[0][idx0]
            orig_idx1 = np.where(mask1)[0][idx1]
            
            # Compute cross-group distances on subsample
            cross_dists = cdist(emb0_sub, emb1_sub, metric="euclidean")
            
            # Max distance pairs (most different)
            max_idx = np.unravel_index(np.argmax(cross_dists), cross_dists.shape)
            max_dist = cross_dists[max_idx]
            
            # Min distance pairs (most similar)
            min_idx = np.unravel_index(np.argmin(cross_dists), cross_dists.shape)
            min_dist = cross_dists[min_idx]
            
            # Find top-N most different pairs (within subsample)
            flat_indices = np.argsort(cross_dists.ravel())[::-1][:n_exemplars]
            max_pairs = np.unravel_index(flat_indices, cross_dists.shape)
            
            exemplars["max_distance_group0"] = orig_idx0[list(max_pairs[0])].tolist()
            exemplars["max_distance_group1"] = orig_idx1[list(max_pairs[1])].tolist()
            
            # Find top-N most similar pairs (within subsample)
            flat_indices = np.argsort(cross_dists.ravel())[:n_exemplars]
            min_pairs = np.unravel_index(flat_indices, cross_dists.shape)
            
            exemplars["min_distance_group0"] = orig_idx0[list(min_pairs[0])].tolist()
            exemplars["min_distance_group1"] = orig_idx1[list(min_pairs[1])].tolist()
            
            print(f"  Max cross-group distance: {max_dist:.4f}")
            print(f"  Min cross-group distance: {min_dist:.4f}")
            print(f"  Found {n_exemplars} extreme pairs for each category")
        
        elif strategy == "boundary" and len(unique_labels) == 2:
            # Spectra near decision boundary (midpoint between centroids)
            label0, label1 = unique_labels
            mask0 = labels == label0
            mask1 = labels == label1
            
            centroid0 = embeddings[mask0].mean(axis=0)
            centroid1 = embeddings[mask1].mean(axis=0)
            boundary = (centroid0 + centroid1) / 2
            
            # Find spectra closest to boundary
            dists = np.linalg.norm(embeddings - boundary, axis=1)
            closest_indices = np.argsort(dists)[:n_exemplars * 2]
            
            exemplars["boundary_spectra"] = closest_indices.tolist()
            
            print(f"  Found {len(closest_indices)} spectra near decision boundary")
        
        else:
            raise ValueError(f"Unsupported strategy or # of groups: {strategy}")
        
        print(f"Exemplar discovery complete")
        
        return exemplars
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "pca",
        label_names: Dict[int, str] = None,
        save_path: Optional[str] = None,
        title: str = "Embedding Visualization"
    ):
        """
        Visualize embeddings using dimensionality reduction (Module 7.2 & 8).
        
        Args:
            embeddings: (N, d) embeddings
            labels: (N,) group labels
            method: Reduction method ("pca", "tsne", "umap")
            label_names: Optional mapping {0: "HCC", 1: "Cirrhosis"}
            save_path: Optional path to save figure
            title: Plot title
        """
        print(f"\n[Module 8] Visualizing embeddings using {method.upper()}")
        
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            print(f"  Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
            
        elif method == "tsne":
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = reducer.fit_transform(embeddings)
            
        elif method == "umap":
            if not UMAP_AVAILABLE:
                print(" UMAP not available, falling back to PCA")
                return self.visualize_embeddings(embeddings, labels, "pca", label_names, save_path, title)
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[label] if label_names else f"Group {label}"
            
            plt.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[colors[i]],
                label=name,
                alpha=0.6,
                s=50,
                edgecolors='k',
                linewidths=0.5
            )
        
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        print(f" Visualization complete")
    
    def plot_distance_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: Dict[int, str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot distributions of intra-group vs inter-group distances (Module 8).
        
        Args:
            embeddings: (N, d) embeddings
            labels: (N,) group labels
            label_names: Optional mapping {0: "HCC", 1: "Cirrhosis"}
            save_path: Optional path to save figure
        """
        print(f"\n[Module 8] Plotting distance distributions")
        
        unique_labels = np.unique(labels)
        
        if len(unique_labels) != 2:
            print("  Distance distribution plot requires exactly 2 groups")
            return
        
        label0, label1 = unique_labels
        mask0 = labels == label0
        mask1 = labels == label1
        
        name0 = label_names[label0] if label_names else f"Group {label0}"
        name1 = label_names[label1] if label_names else f"Group {label1}"
        
        # Compute distances
        dist_matrix = squareform(pdist(embeddings, metric="euclidean"))
        
        intra0 = dist_matrix[np.ix_(mask0, mask0)][np.triu_indices(mask0.sum(), k=1)]
        intra1 = dist_matrix[np.ix_(mask1, mask1)][np.triu_indices(mask1.sum(), k=1)]
        inter = dist_matrix[np.ix_(mask0, mask1)].flatten()
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(intra0, bins=50, alpha=0.5, label=f"Intra-{name0}", color='blue', density=True)
        plt.hist(intra1, bins=50, alpha=0.5, label=f"Intra-{name1}", color='green', density=True)
        plt.hist(inter, bins=50, alpha=0.5, label=f"Inter-group", color='red', density=True)
        plt.xlabel("Euclidean Distance", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title("Distance Distributions", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        data = [intra0, intra1, inter]
        labels_box = [f"Intra-{name0}", f"Intra-{name1}", "Inter-group"]
        bp = plt.boxplot(data, labels=labels_box, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Euclidean Distance", fontsize=12)
        plt.title("Distance Comparison", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        print(f" Distance distribution plot complete")
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        save_path: str
    ):
        """
        Generate analysis report (Module 8).
        
        Args:
            metrics: Dictionary of computed metrics
            save_path: Path to save report
        """
        print(f"\n[Module 8] Generating analysis report")
        
        report = []
        report.append("=" * 80)
        report.append("MS-Spectral-Foundation: Embedding Analysis Report")
        report.append("Modules 7 & 8: Distribution Analysis and Representation Evaluation")
        report.append("=" * 80)
        report.append("")
        
        report.append("## Distance Metrics")
        report.append("-" * 80)
        for key, value in metrics.items():
            if "mean" in key or "std" in key or "ratio" in key:
                report.append(f"  {key:40s}: {value:.6f}")
        report.append("")
        
        report.append("## Statistical Tests")
        report.append("-" * 80)
        for key, value in metrics.items():
            if "statistic" in key or "p_value" in key:
                if "p_value" in key:
                    report.append(f"  {key:40s}: {value:.4e}")
                else:
                    report.append(f"  {key:40s}: {value:.6f}")
        report.append("")
        
        report.append("## Interpretation")
        report.append("-" * 80)
        
        if "separation_ratio" in metrics:
            ratio = metrics["separation_ratio"]
            if ratio > 1.2:
                interpretation = "Strong separation: Inter-group distances significantly larger than intra-group"
            elif ratio > 1.0:
                interpretation = "Moderate separation: Some distinguishable structure between groups"
            else:
                interpretation = "Weak separation: Groups are not well separated in embedding space"
            report.append(f"  {interpretation}")
        
        if "mannwhitney_p_value" in metrics:
            pval = metrics["mannwhitney_p_value"]
            if pval < 0.001:
                sig = "Highly significant (p < 0.001)"
            elif pval < 0.05:
                sig = "Significant (p < 0.05)"
            else:
                sig = "Not significant (p >= 0.05)"
            report.append(f"  Statistical significance: {sig}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(save_path, "w") as f:
            f.write(report_text)
        
        print(f"Report saved to: {save_path}")
        print("\n" + report_text)


def create_sample_metadata(
    mgf_paths: List[str],
    group_labels: List[str]
) -> pd.DataFrame:
    """
    Helper function to create metadata DataFrame from MGF files.
    
    Args:
        mgf_paths: List of MGF file paths
        group_labels: List of group labels (e.g., ["HCC", "Cirrhosis", ...])
    
    Returns:
        DataFrame with columns: sample_id, mgf_path, group
    """
    metadata = []
    
    for mgf_path, group in zip(mgf_paths, group_labels):
        # Extract sample ID from filename
        sample_id = Path(mgf_path).stem
        
        metadata.append({
            "sample_id": sample_id,
            "mgf_path": mgf_path,
            "group": group
        })
    
    return pd.DataFrame(metadata)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MS-Spectral-Foundation: Embedding Analysis (Modules 7 & 8)")
    print("="*80)
    print("\nThis module implements:")
    print("  - Module 7: Embedding-space distribution analysis")
    print("  - Module 8: Non-supervised representation evaluation")
    print("\nUsage example:")
    print("""
    # 1. Initialize analyzer with pretrained model
    analyzer = EmbeddingAnalyzer(
        model_path="outputs/model_ssl_v4_epoch099.pt",
        device="cuda"
    )
    
    # 2. Load dataset
    dataset = SpectrumDataset(
        mgf_path="path/to/data.mgf",
        max_peaks=150,
        mz_min=50.0,
        mz_max=2500.0,
        bin_size=0.5
    )
    
    # 3. Extract embeddings (Module 6)
    results = analyzer.extract_embeddings(dataset, batch_size=64)
    embeddings = results["embeddings"]
    
    # 4. Create sample metadata (group labels)
    # For real analysis, load from file or MGF metadata
    sample_ids = np.array([f"sample_{i//100}" for i in range(len(embeddings))])
    labels = np.array([0 if i < len(embeddings)//2 else 1 for i in range(len(embeddings))])
    label_names = {0: "HCC", 1: "Cirrhosis"}
    
    # 5. Sample-level aggregation (Module 7.1)
    sample_emb, unique_samples = analyzer.aggregate_to_sample_level(
        embeddings, sample_ids, method="mean"
    )
    sample_labels = np.array([labels[sample_ids == s][0] for s in unique_samples])
    
    # 6. Distribution analysis (Module 7.2)
    metrics = analyzer.analyze_embedding_distributions(
        sample_emb, sample_labels, label_names
    )
    
    # 7. Exemplar discovery (Module 7.3)
    exemplars = analyzer.find_exemplar_spectra(
        embeddings, labels, dataset, n_exemplars=5, strategy="extremes"
    )
    
    # 8. Visualization (Module 8)
    analyzer.visualize_embeddings(
        sample_emb, sample_labels, method="pca",
        label_names=label_names,
        save_path="logs/embeddings_pca.png"
    )
    
    analyzer.plot_distance_distributions(
        sample_emb, sample_labels,
        label_names=label_names,
        save_path="logs/distance_distributions.png"
    )
    
    # 9. Generate report (Module 8)
    analyzer.generate_report(metrics, save_path="logs/analysis_report.txt")
    """)
    print("\n" + "="*80)
