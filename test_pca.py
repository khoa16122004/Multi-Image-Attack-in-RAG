import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from lvlm_models.llava_ import LLava
from util import DataLoader
from tqdm import tqdm

# --- Load data ---
result_clean_dir = "result_usingquery=0_clip"
loader = DataLoader(retri_dir=result_clean_dir)
sample_path = "run.txt"

with open(sample_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]

# --- Load model ---
# model = LLava(
#     pretrained="llava-next-interleave-qwen-7b",
#     model_name="llava_qwen",
# )
model = LLava(
    pretrained="llava-onevision-qwen2-7b-ov",
    model_name="llava_qwen",
)

def apply_pca_colormap_batch(patch_features, n_components=3):
    """
    Apply PCA to batch of patch features and convert to RGB colormap
    
    Args:
        patch_features: numpy array of shape (B, 729, D) where B is batch size, D is feature dimension
        n_components: number of PCA components to use (typically 3 for RGB)
    
    Returns:
        pca_features: numpy array of shape (B, 729, 3) with PCA components
    """
    B, num_patches, D = patch_features.shape
    
    # Reshape to (B*729, D) for PCA fitting
    features_reshaped = patch_features.reshape(-1, D)  # (B*729, D)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_reshaped)  # (B*729, 3)
    
    print(f"PCA input shape: {features_reshaped.shape}")
    print(f"PCA output shape: {pca_features.shape}")
    
    for i in range(n_components):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    
    # Reshape back to (B, 729, 3)
    pca_features = pca_features.reshape(B, num_patches, n_components)
    
    print(f"Final PCA shape: {pca_features.shape}")
    
    return pca_features

def create_pca_visualization_batch(retri_imgs, patch_feats_batch, sample_id, grid_size=(27, 27), 
                                  bg_threshold=15, save_dir="pca_visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = len(retri_imgs)
    
    n_cols = num_images  # All images in one row
    
    fig, axes = plt.subplots(2, n_cols, figsize=(3*n_cols, 6))
    
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    print("Patch_features batch shape:", patch_feats_batch.shape)
    pca_features_batch = apply_pca_colormap_batch(patch_feats_batch)  # Shape: (B, 729, 3)
    
    for img_idx in range(num_images):
        col = img_idx
        
        img = retri_imgs[img_idx]
        pca_features = pca_features_batch[img_idx]  # Shape: (729, 3)
        
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img_np = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img.cpu().numpy()
        else:
            img_np = np.array(img)
        
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        if n_cols == 1:
            axes[0].imshow(img_np)
            axes[0].axis('off')
            axes[0].set_title('Original Image', fontsize=12)
        else:
            axes[0, col].imshow(img_np)
            axes[0, col].axis('off')
            axes[0, col].set_title(f'Image {img_idx+1}', fontsize=10)
        
        h, w = grid_size
        pca_rgb = pca_features.reshape(h, w, 3)
        
        threshold = np.percentile(pca_features[:, 0], bg_threshold)
        mask = pca_features[:, 0] > threshold
        mask = mask.reshape(h, w)
        
        # Create visualization with black background
        vis_img = np.zeros((h, w, 3))
        vis_img[mask] = pca_rgb[mask]
        
        # Plot PCA visualization in bottom row
        if n_cols == 1:
            axes[1].imshow(vis_img)
            axes[1].axis('off')
            axes[1].set_title('PCA Visualization', fontsize=12)
        else:
            axes[1, col].imshow(vis_img)
            axes[1, col].axis('off')
            axes[1, col].set_title(f'PCA {img_idx+1}', fontsize=10)
    
    plt.suptitle(f'Sample {sample_id}: PCA Visualization of Patch Features ({num_images} images)', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'sample_{sample_id}_pca_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Saved visualization for sample {sample_id} to {save_path} (threshold={bg_threshold}%)")

# --- Main execution ---
if __name__ == "__main__":
    # Create output directory
    output_dir = "pca_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total samples to process: {len(sample_ids)}")
    print(f"Sample IDs: {sample_ids}")
    
    processed_count = 0
    failed_count = 0
    threshold = 0
    
    for i in tqdm(sample_ids, desc="Processing all samples"):
        try:
            print(f"\n[{processed_count + failed_count + 1}/{len(sample_ids)}] Processing sample {i}...")
            
            question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(i)
            
            patch_feats = model.extract_patch_features(retri_imgs)  # Shape: (num_images, 729, D)
            
            if isinstance(patch_feats, torch.Tensor):
                patch_feats = patch_feats.cpu().numpy()
            
            print(f"  - Patch features shape: {patch_feats.shape}")
            print(f"  - Number of retrieved images: {len(retri_imgs)}")
            
            create_pca_visualization_batch(retri_imgs, patch_feats, i, 
                                         bg_threshold=threshold, save_dir=output_dir)  # Adjustable threshold
            
            processed_count += 1
            print(f"  ✓ Successfully processed sample {i}")
            
        except Exception as e:
            failed_count += 1
            print(f"  ✗ Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error info to file
            error_file = os.path.join(output_dir, f"error_sample_{i}.txt")
            with open(error_file, "w") as f:
                f.write(f"Error processing sample {i}:\n")
                f.write(f"{str(e)}\n\n")
                f.write(traceback.format_exc())
            
            continue
    
    print(f"\n" + "="*50)
    print(f"PROCESSING COMPLETE!")
    print(f"Total samples: {len(sample_ids)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"="*50)

# Alternative function for processing single sample
def process_single_sample(sample_id, loader, model, bg_threshold=20, output_dir="pca_visualizations"):
    """
    Process a single sample and create PCA visualization
    
    Args:
        sample_id: sample identifier
        loader: data loader
        model: model for feature extraction
        bg_threshold: percentile threshold for background removal
        output_dir: output directory
    """
    try:
        question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)
        
        # Extract patch features
        patch_feats = model.extract_patch_features(retri_imgs)  # Shape: (num_images, 729, D)
        
        if isinstance(patch_feats, torch.Tensor):
            patch_feats = patch_feats.cpu().numpy()
        
        # Create visualization
        create_pca_visualization_batch(retri_imgs, patch_feats, sample_id, 
                                     bg_threshold=bg_threshold, save_dir=output_dir)
        
        return True
        
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return False

# Example usage for specific samples with different thresholds
def process_selected_samples(sample_list, loader, model, bg_threshold=20, output_dir="pca_visualizations"):
    """
    Process only selected samples
    
    Args:
        sample_list: list of sample IDs to process
        loader: data loader
        model: model for feature extraction
        bg_threshold: percentile threshold for background removal
        output_dir: output directory
    """
    for sample_id in tqdm(sample_list, desc="Processing selected samples"):
        success = process_single_sample(sample_id, loader, model, bg_threshold, output_dir)
        if success:
            print(f"Successfully processed sample {sample_id}")
        else:
            print(f"Failed to process sample {sample_id}")

# Function to experiment with different thresholds
def experiment_thresholds(sample_id, loader, model, thresholds=[5, 10, 15, 20, 25, 30], 
                         output_dir="pca_experiments"):
    """
    Experiment with different background thresholds for a single sample
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)
        patch_feats = model.extract_patch_features(retri_imgs)
        
        if isinstance(patch_feats, torch.Tensor):
            patch_feats = patch_feats.cpu().numpy()
        
        for threshold in thresholds:
            print(f"Testing threshold {threshold}% for sample {sample_id}")
            
            # Create subfolder for this threshold
            thresh_dir = os.path.join(output_dir, f"threshold_{threshold}")
            create_pca_visualization_batch(retri_imgs, patch_feats, sample_id, 
                                         bg_threshold=threshold, save_dir=thresh_dir)
            
    except Exception as e:
        print(f"Error in threshold experiment for sample {sample_id}: {e}")

# Uncomment to process only first 5 samples with custom threshold
# process_selected_samples(sample_ids[:5], loader, model, bg_threshold=25)

# Uncomment to experiment with different thresholds on one sample
# experiment_thresholds(sample_ids[0], loader, model)