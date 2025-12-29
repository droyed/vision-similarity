"""
Feature Similarity Heatmap Generator

This script generates visual similarity heatmaps comparing images using various
vision transformer models and feature extraction strategies.

Overview:
---------
The script systematically evaluates 13 different pre-trained vision models
(DINOv3, DINOv2, ViT, BEiT) across 4 feature extraction strategies to create
similarity matrices. For each model-strategy combination, it:

1. Loads images from the 'assets' directory
2. Extracts features using the specified model and strategy
3. Computes pairwise similarity between all images
4. Generates and saves a heatmap visualization

Models Tested:
--------------
- DINOv3: vits16, vitb16, vitl16, vit7b16 (pretrained on lvd1689m)
- DINOv2: small, base, large, giant
- Google ViT: base-patch16-224, large-patch16-224
- Microsoft BEiT: base-patch16-224, large-patch16-224

Feature Extraction Strategies:
------------------------------
- mean_patches: Average of all patch tokens (excluding CLS token)
- cls: Uses only the CLS (classification) token
- mean_all: Average of all tokens including CLS
- pooler: Uses model's pooler output (if available)

Output:
-------
Saves heatmap visualizations to 'output_heatmaps/' directory with naming
pattern: {model_name}__{strategy}.png

Total outputs: 52 heatmaps (13 models × 4 strategies)
"""

import os
from vision_similarity import FeatureSimilarity
from vision_similarity.utils import list_image_files


models = [
    'facebook/dinov3-vits16-pretrain-lvd1689m',
    'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    'facebook/dinov2-small',
    'facebook/dinov2-base',
    'facebook/dinov2-large',
    'facebook/dinov2-giant',
    'google/vit-base-patch16-224',
    'google/vit-large-patch16-224',
    'microsoft/beit-base-patch16-224',
    'microsoft/beit-large-patch16-224',
 ]

# Use smaller models for testing
models = [
    'facebook/dinov3-vits16-pretrain-lvd1689m',
    'facebook/dinov2-small',
    'google/vit-base-patch16-224',
    'microsoft/beit-base-patch16-224',
 ]

strategies = ["mean_patches", "cls", "mean_all", "pooler"]


imgpaths = list_image_files('assets', sort=True)

outputdir = 'output_heatmaps'
os.makedirs(outputdir, exist_ok=True)

for pretrained_model_name in models:
    print(f'pretrained_model_name = {pretrained_model_name}')
    
    # Get similarity matrix directly from image store for all strategies
    for strategy in strategies:
        savepath = f'{outputdir}/{pretrained_model_name.replace("/", "__")}_{strategy}.png'
        canvas = FeatureSimilarity.from_image_sets(
            pretrained_model_name=pretrained_model_name,
            image_paths_set1=imgpaths,
            strategy=strategy,
        ).plot(title=f'Strategy={strategy}', savepath=savepath)
