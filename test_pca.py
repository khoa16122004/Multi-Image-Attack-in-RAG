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
model = LLava(
    pretrained="llava-next-interleave-qwen-7b",
    model_name="llava_qwen",
)

def apply_pca_colormap(patch_features, n_components=3):
    """
    Apply PCA to patch features and convert to RGB colormap
    
    Args:
        patch_features: numpy array of shape (729, D) where D is feature dimension
        n_components: number of PCA components to use (typically 3 for RGB)
    
    Returns:
        pca_features: numpy array of shape (729, 3) with PCA components
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(patch_features)  # (729, 3)
    
    # Normalize each component to [0, 1] range
    for i in range(n_components):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    
    return pca_features

def create_pca_visualization_batch(retri_imgs, patch_feats_batch, sample_id, grid_size=(27, 27), save_dir="pca_visualizations"):
    """
    Create PCA visualization for a batch of images
    
    Args:
        retri_imgs: list of images
        patch_feats_batch: numpy array of shape (num_images, 729, D)
        sample_id: sample identifier
        grid_size: tuple of (height, width) for patch grid (default 27x27 = 729)
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = len(retri_imgs)
    
    # Create layout
    n_cols = min(4, num_images)  # Max 4 columns
    n_rows = (num_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Handle different subplot configurations
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(2, 1)
    elif n_rows == 1:
        axes = axes.reshape(2, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for img_idx in range(num_images):
        row = (img_idx // n_cols) * 2
        col = img_idx % n_cols
        
        # Get current image and patch features
        img = retri_imgs[img_idx]
        patch_feats = patch_feats_batch[img_idx]  # Shape: (729, D)
        
        # Convert image to numpy if needed
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img_np = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img.cpu().numpy()
        else:
            img_np = np.array(img)
        
        # Ensure image is in [0, 1] range for display
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Plot original image
        if n_rows == 1 and n_cols == 1:
            axes[0].imshow(img_np)
            axes[0].axis('off')
            axes[0].set_title('Original Image', fontsize=12)
        else:
            axes[row, col].imshow(img_np)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Image {img_idx+1}', fontsize=10)
        
        # Apply PCA to patch features
        pca_features = apply_pca_colormap(patch_feats)  # Shape: (729, 3)
        
        # Reshape to grid
        h, w = grid_size
        pca_rgb = pca_features.reshape(h, w, 3)
        
        # Background removal using first PCA component
        threshold = np.percentile(pca_features[:, 0], 15)  # Remove bottom 15%
        mask = pca_features[:, 0] > threshold
        mask = mask.reshape(h, w)
        
        # Create visualization with black background
        vis_img = np.zeros((h, w, 3))
        vis_img[mask] = pca_rgb[mask]
        
        # Plot PCA visualization
        if n_rows == 1 and n_cols == 1:
            axes[1].imshow(vis_img)
            axes[1].axis('off')
            axes[1].set_title('PCA Visualization', fontsize=12)
        else:
            axes[row + 1, col].imshow(vis_img)
            axes[row + 1, col].axis('off')
            axes[row + 1, col].set_title(f'PCA {img_idx+1}', fontsize=10)
    
    # Hide unused subplots
    for img_idx in range(num_images, n_rows * n_cols):
        row = (img_idx // n_cols) * 2
        col = img_idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
    
    plt.suptitle(f'Sample {sample_id}: PCA Visualization of Patch Features ({num_images} images)', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'sample_{sample_id}_pca_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Saved visualization for sample {sample_id} to {save_path}")

# --- Main execution ---
if __name__ == "__main__":
    # Create output directory
    output_dir = "pca_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total samples to process: {len(sample_ids)}")
    print(f"Sample IDs: {sample_ids}")
    
    # Process each sample individually
    processed_count = 0
    failed_count = 0
    
    for i in tqdm(sample_ids, desc="Processing all samples"):
        try:
            print(f"\n[{processed_count + failed_count + 1}/{len(sample_ids)}] Processing sample {i}...")
            
            # Load data for this sample
            question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(i)
            
            # Extract patch features for all images in batch
            patch_feats = model.extract_patch_features(retri_imgs)  # Shape: (num_images, 729, D)
            
            # Convert to numpy if tensor
            if isinstance(patch_feats, torch.Tensor):
                patch_feats = patch_feats.cpu().numpy()
            
            print(f"  - Patch features shape: {patch_feats.shape}")
            print(f"  - Number of retrieved images: {len(retri_imgs)}")
            
            # Create and save visualization
            create_pca_visualization_batch(retri_imgs, patch_feats, i, save_dir=output_dir)
            
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
def process_single_sample(sample_id, loader, model, output_dir="pca_visualizations"):
    """
    Process a single sample and create PCA visualization
    """
    try:
        question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)
        
        # Extract patch features
        patch_feats = model.extract_patch_features(retri_imgs)  # Shape: (num_images, 729, D)
        
        if isinstance(patch_feats, torch.Tensor):
            patch_feats = patch_feats.cpu().numpy()
        
        # Create visualization
        create_pca_visualization_batch(retri_imgs, patch_feats, sample_id, save_dir=output_dir)
        
        return True
        
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return False

# Example usage for specific samples
def process_selected_samples(sample_list, loader, model, output_dir="pca_visualizations"):
    """
    Process only selected samples
    """
    for sample_id in tqdm(sample_list, desc="Processing selected samples"):
        success = process_single_sample(sample_id, loader, model, output_dir)
        if success:
            print(f"Successfully processed sample {sample_id}")
        else:
            print(f"Failed to process sample {sample_id}")

# Uncomment to process only first 5 samples
process_selected_samples(sample_ids, loader, model)