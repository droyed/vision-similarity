from vision_similarity import ImageFeatureStore, load_vision_model, make_deterministic, extract_image_features, cosine_similarity_matrix, FeatureSimilarity
from vision_similarity.utils import list_image_files
import numpy as np
import os


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

make_deterministic(42)

imgpaths = list_image_files('assets', sort=True)

## Usage of ImageFeatureStore class, load_vision_model function
### Usage 1: Compute features for a set of images
print("\n[Usage 1] : Compute features for a set of images")
store = ImageFeatureStore.from_images(image_paths=imgpaths, pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
features = store.features
image_paths = store.image_paths
model_name = store.model_name
file_version = store.FILE_VERSION
print(f"file version: {file_version}")
print(f"features shape: {features.shape}")
print(f"image paths: {image_paths}")
print(f"model name: {model_name}")

model, processor = load_vision_model(pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
features_extracted = extract_image_features(image_paths=imgpaths, model=model, processor=processor, batch_size=5)
assert (np.allclose(features, features_extracted, atol=1e-5)), "Features are not equal"
print("Features are equal")

## use different strategy for extraction
features_extracted_cls = extract_image_features(image_paths=imgpaths, model=model, processor=processor, batch_size=5, strategy="cls")
features_extracted_mean_all = extract_image_features(image_paths=imgpaths, model=model, processor=processor, batch_size=5, strategy="mean_all")
features_extracted_pooler = extract_image_features(image_paths=imgpaths, model=model, processor=processor, batch_size=5, strategy="pooler")
    
### Usage 2: Compute features for a set of images given model and processor
print("\n[Usage 2] : Compute features for a set of images given model and processor")   
model, processor = load_vision_model(pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
store = ImageFeatureStore.from_images_with_model(image_paths=imgpaths, model=model, processor=processor, model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
features = store.features
print(f"features shape: {features.shape}")

assert (np.allclose(features, store.features)), "Features are not equal"
print("Features are equal")

### Usage 3: Test saving and loading features to a file and equality dunder method
print("\n[Usage 3] : Test saving and loading features to a file")
store.save_npz('features.npz')
store_loaded = ImageFeatureStore.from_npz('features.npz')
assert (store == store_loaded), "Stores are not equal"
print("Stores are equal")

### Usage 4: Dunder method usage examples for ImageFeatureStore
print("\n[Usage 4] : Dunder method usage examples for ImageFeatureStore")
# __str__ and __repr__ usage
print("str(store):", str(store))
print("repr(store):", repr(store))

# __len__ usage
print("Number of images in store:", len(store))

# __getitem__ usage
first_image_path, first_feature = store[0]
print(f"First image path: {first_image_path}")
print(f"First feature vector shape: {first_feature.shape}")

# __iter__ usage
for idx, (image_path, feature) in enumerate(store):
    print(f"Image {idx}: {image_path} | feature shape: {feature.shape}")

# __eq__ usage
store_copy = ImageFeatureStore(
    features=store.features.copy(),
    image_paths=list(store.image_paths),
    model_name=store.model_name
)
print("store == store_copy:", store == store_copy)

### Usage 5: Compute similarity matrix using FeatureSimilarity class
print("\n[Usage 5] : Compute similarity matrix using FeatureSimilarity class")
heatmap_savepath = 'heatmap.png'
pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m"

sim = FeatureSimilarity.from_image_sets(
    pretrained_model_name=pretrained_model_name,
    image_paths_set1=imgpaths,
    batch_size=2,
    strategy='cls',
)
sim.plot(savepath=heatmap_savepath)

### Usage 6: Compute similarity matrix using cosine_similarity_matrix function
print("\n[Usage 6] : Compute similarity matrix using cosine_similarity_matrix function")
sim = cosine_similarity_matrix(features, features)
print(f"cosine similarity matrix shape: {sim.shape}")
assert (np.allclose(np.diag(sim), 1.0)), "Cosine similarity matrix diagonal is not equal to 1.0"
print("Cosine similarity matrix diagonal is equal to 1.0")


# Cleanup with user confirmation    
#confirm = input("Do you want to remove the temporary files? (y/n): ") # commenting for pytests run
confirm = "y"

if confirm == 'y':
    print("\n[Cleanup] : Remove temporary files")
    if os.path.exists('heatmap.png'):
        os.remove('heatmap.png')
    if os.path.exists('features.npz'):
        os.remove('features.npz')
    print("\n[Cleanup] : Done")
else:
    print("\n[Cleanup] : Skipping removal of temporary files")
    print("\n[Cleanup] : Done")