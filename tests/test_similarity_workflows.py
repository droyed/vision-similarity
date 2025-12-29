from vision_similarity import ImageFeatureStore, FeatureSimilarity, make_deterministic
from vision_similarity.utils import list_image_files

make_deterministic(42)

imgpaths = list_image_files('assets', sort=True)

# Workflow 1: Use ImageFeatureStore to set up the feature store and plot the similarity matrix
# Setup feature store instance
store = ImageFeatureStore.from_images(image_paths=imgpaths, pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m", strategy="cls")

# Plot it with FeatureSimilarity
FeatureSimilarity(store).plot(title="Similarity Matrix - Workflow 1", block=True)

# Workflow 2: Get similarity matrix directly from image sets
sim_matrix = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    image_paths_set1=imgpaths,
    strategy="cls",
)
plot_image = sim_matrix.plot(title="Similarity Matrix - Workflow 2", block=True)

# Workflow 3: Get similarity matrix plot directly from image sets
sim_matrix_plot = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    image_paths_set1=imgpaths,
    strategy="cls",
).plot(title="Similarity Matrix - Workflow 3")
