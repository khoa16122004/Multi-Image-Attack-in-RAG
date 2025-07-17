import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from lvlm_models.llava_ import LLava
from util import DataLoader
from tqdm import tqdm
from PIL import Image

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

def reshape_to_spatial(features, patch_size=16, img_size=224):
    """
    Reshape flattened patch features back to spatial format
    features: (num_patches, feature_dim) - assuming 729 patches for 27x27 grid
    """
    grid_size = int(np.sqrt(features.shape[0]))  # 27 for 729 patches
    feature_dim = features.shape[1]
    
    # Reshape to (grid_size, grid_size, feature_dim)
    spatial_features = features.reshape(grid_size, grid_size, feature_dim)
    return spatial_features

def compute_pca_visualization(features_list, patch_size=16, n_components=3):
    """
    Compute PCA on concatenated features from all images
    features_list: list of (num_patches, feature_dim) arrays
    """
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(all_features)
    
    # Normalize PCA features to [0, 1] for RGB visualization
    pca_features_norm = []
    for i in range(n_components):
        component = pca_features[:, i]
        component_norm = (component - component.min()) / (component.max() - component.min())
        pca_features_norm.append(component_norm)
    
    pca_features_norm = np.stack(pca_features_norm, axis=1)
    
    # Split back into individual images
    pca_images = []
    start_idx = 0
    for features in features_list:
        end_idx = start_idx + features.shape[0]
        pca_img = pca_features_norm[start_idx:end_idx]
        pca_images.append(pca_img)
        start_idx = end_idx
    
    return pca_images, pca

def upsample_pca_image(pca_spatial, mask, patch_size=16, img_size=224):
    """
    Upsample PCA spatial features to image size
    """
    h, w = pca_spatial.shape[:2]
    pca_image = np.zeros((img_size, img_size, 3), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                # Calculate patch boundaries
                y0 = i * patch_size
                y1 = min((i + 1) * patch_size, img_size)
                x0 = j * patch_size
                x1 = min((j + 1) * patch_size, img_size)
                
                pca_image[y0:y1, x0:x1] = pca_spatial[i, j]
    
    return (pca_image * 255).clip(0, 255).astype(np.uint8)

def create_pca_visualization(patch_features_list, imgs, patch_size=16, threshold_percentile=25):
    """
    Create PCA visualization for LLaVA patch features
    patch_features_list: list of (729, D) arrays from model.extract_patch_features
    imgs: list of PIL images
    """
    # Convert torch tensors to numpy if needed
    features_list = []
    for features in patch_features_list:
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        features_list.append(features)
    
    # Compute PCA
    pca_images, pca = compute_pca_visualization(features_list, patch_size, n_components=3)
    
    upsampled_imgs = []
    
    # Create visualization
    fig, axes = plt.subplots(2, len(imgs), figsize=(len(imgs) * 3, 6))
    
    # Handle single image case
    if len(imgs) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (img, pca_feat) in enumerate(zip(imgs, pca_images)):
        # Reshape to spatial format
        pca_spatial = reshape_to_spatial(pca_feat, patch_size)
        
        # Create mask based on first component
        first_component = pca_spatial[:, :, 0]
        threshold = np.percentile(first_component, threshold_percentile)
        mask = first_component > threshold
        
        # Apply mask
        pca_spatial_masked = pca_spatial.copy()
        pca_spatial_masked[~mask] = 0
        
        # Upsample to image size
        upsampled_img = upsample_pca_image(pca_spatial_masked, mask, patch_size)
        upsampled_imgs.append(upsampled_img)
        
        # Plot original image (resize to 224x224)
        if hasattr(img, 'resize'):
            img_resized = img.resize((224, 224))
        else:
            img_resized = img
        
        axes[0, i].imshow(img_resized)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Plot PCA RGB
        axes[1, i].imshow(upsampled_img)
        axes[1, i].set_title(f'PCA RGB {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    return upsampled_imgs, pca

# --- Main visualization loop ---
for i in tqdm(sample_ids):
    question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(i)
    
    # Extract patch features - returns list of (729, D) tensors
    patch_feats = model.extract_patch_features(retri_imgs)
    
    # Create PCA visualization
    print(f"\nProcessing sample {i}:")
    print(f"Question: {question}")
    print(f"Number of retrieved images: {len(retri_imgs)}")
    
    try:
        upsampled_imgs, pca = create_pca_visualization(
            patch_feats, 
            retri_imgs, 
            patch_size=16,  # Adjust based on your model's patch size
            threshold_percentile=25
        )
        
        # Optional: Save results
        save_dir = f"pca_results/sample_{i}"
        os.makedirs(save_dir, exist_ok=True)
        
        for j, upsampled_img in enumerate(upsampled_imgs):
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(retri_imgs[j].resize((224, 224)))
            plt.title(f'Original Image {j+1}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(upsampled_img)
            plt.title(f'PCA Visualization {j+1}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/pca_comparison_{j+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue
    
    # Optional: Add pause between samples
    input("Press Enter to continue to next sample...")

print("PCA visualization completed!")