import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from typing import List, Optional, Tuple, Iterator
from PIL import Image
import torch.nn.functional as F
import random


def make_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    
    # Critical for GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_image_features(
    image_paths: List[str],
    model,
    processor,
    batch_size: int = 16,
    cleanup: bool = True,
    strategy: str = "mean_patches",
) -> np.ndarray:
    """
    strategy options:
        - "mean_patches"     : mean of patch tokens (recommended for similarity)
        - "cls"              : CLS token
        - "mean_all"         : mean of CLS + patches
        - "pooler"           : pooler_output (not recommended for similarity)
    """
    all_features = []

    model.eval()
    model_device = next(model.parameters()).device

    with torch.inference_mode():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Load images
            images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))

            # Preprocess
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            outputs = model(**inputs)

            hidden = outputs.last_hidden_state  # (B, 1 + P, D)

            # -----------------------------
            # Feature extraction strategies
            # -----------------------------
            if strategy == "mean_patches":
                # 1️⃣ Mean of patch tokens only (best for similarity)
                features = hidden[:, 1:, :].mean(dim=1)

            elif strategy == "cls":
                # 2️⃣ CLS token
                features = hidden[:, 0, :]

            elif strategy == "mean_all":
                # 3️⃣ Mean of CLS + patch tokens
                features = hidden.mean(dim=1)

            elif strategy == "pooler":
                # 4️⃣ Pooler output (generally avoid for similarity)
                if outputs.pooler_output is None:
                    raise ValueError("Model does not provide pooler_output")
                features = outputs.pooler_output

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Move to CPU
            all_features.append(features.cpu().numpy())

            # Cleanup
            if cleanup:
                del inputs, outputs, hidden, features
                torch.cuda.empty_cache()

    return np.vstack(all_features)

def load_vision_model(pretrained_model_name: str, device_map="auto"):
    """
    Load a pretrained vision model and its image processor from HuggingFace.
    
    Args:
        pretrained_model_name: Name or path of the pretrained model (e.g., "google/vit-base-patch16-224")
        device_map: Device mapping strategy for model loading. Defaults to "auto" which automatically
                   selects the best device. Can be "cpu", "cuda", or a specific device map dict.
    
    Returns:
        Tuple containing:
            - model: The loaded vision model (set to evaluation mode)
            - processor: The corresponding AutoImageProcessor for preprocessing images
    """
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name, 
        use_fast=True,
        )
    model = AutoModel.from_pretrained(
        pretrained_model_name,
        device_map=device_map,
    )
    model.eval()
    return model, processor

def cosine_similarity_matrix(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of features.
    
    Args:
        features1: np.ndarray of shape [n_images1, hidden_dim] containing feature vectors
        features2: np.ndarray of shape [n_images2, hidden_dim] containing feature vectors
    
    Returns:
        sim: numpy array of shape [n_images1, n_images2] with cosine similarity values
    """
    features1 = torch.from_numpy(features1)
    features2 = torch.from_numpy(features2)

    # normalize the features
    normalized_features1 = F.normalize(features1, p=2, dim=1)
    normalized_features2 = F.normalize(features2, p=2, dim=1)

    # compute the similarity matrix
    similarity_matrix = torch.mm(normalized_features1, normalized_features2.T)
    sim = similarity_matrix.detach().float().cpu().numpy()

    return sim

class ImageFeatureStore:
    FILE_VERSION = "1.0"

    def __init__(
        self,
        features: np.ndarray,
        image_paths: List[str],
        model_name: Optional[str] = None,
    ):
        assert features.shape[0] == len(image_paths)
        self.features = features
        self.image_paths = image_paths
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Construction from images
    # ------------------------------------------------------------------
    @classmethod
    def from_images(
        cls,
        image_paths: List[str],
        pretrained_model_name: str,
        device_map="auto",
        batch_size: int = 16,
        strategy: str = "mean_patches",
    ):
        model, processor = load_vision_model(
            pretrained_model_name=pretrained_model_name,
            device_map=device_map,
        )
        return cls.from_images_with_model(
            image_paths=image_paths,
            model=model,
            processor=processor,
            model_name=pretrained_model_name,
            batch_size=batch_size,
            strategy=strategy,
        )

    @classmethod
    def from_images_with_model(
        cls,
        image_paths: List[str],
        model,
        processor,
        model_name: Optional[str] = None,
        batch_size: int = 16,
        strategy: str = "mean_patches",
    ):

        model.eval()

        features = extract_image_features(
            image_paths=image_paths,
            model=model,
            processor=processor,
            batch_size=batch_size,
            strategy=strategy,
        )

        features = features.astype("float32")

        return cls(
            features=features,
            image_paths=image_paths,
            model_name=model_name,
        )


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_npz(self, filepath: str):
        """
        Save features and image paths to a compressed NPZ file.
        Enforces float32 features and stores metadata for future safety.
        """
        np.savez_compressed(
            filepath,
            features=self.features.astype("float32"),
            image_paths=np.array(self.image_paths),
            version=self.FILE_VERSION,
            model_name=self.model_name,
        )

    @classmethod
    def from_npz(cls, filepath: str):
        """
        Load features and image paths from an NPZ file and reconstruct
        an ImageFeatureStore instance.
        """
        data = np.load(filepath, allow_pickle=True)

        version = str(data.get("version", "unknown"))
        if version != cls.FILE_VERSION:
            raise ValueError(
                f"Unsupported file version: {version} "
                f"(expected {cls.FILE_VERSION})"
            )

        features = data["features"]
        image_paths = data["image_paths"].tolist()
        model_name = data.get("model_name", None)

        return cls(
            features=features,
            image_paths=image_paths,
            model_name=str(model_name) if model_name is not None else None,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            f"ImageFeatureStore("
            f"N={len(self)}, "
            f"dim={self.features.shape[1]}, "
            f"dtype={self.features.dtype}, "
            f"model={self.model_name})"
        )

    def __repr__(self) -> str:
        return (
            f"ImageFeatureStore("
            f"features_shape={self.features.shape}, "
            f"image_paths={len(self.image_paths)}, "
            f"model_name={self.model_name!r})"
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        return self.image_paths[idx], self.features[idx]

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ImageFeatureStore):
            return False
        return (
            self.model_name == other.model_name
            and self.image_paths == other.image_paths
            and np.array_equal(self.features, other.features)
        )

class FeatureSimilarity:
    def __init__(
        self,
        store1,
        store2: Optional = None,
        metric: str = "cosine",
    ):
        self.store1 = store1
        self.store2 = store1 if store2 is None else store2
        self.metric = metric
        self._similarity_matrix: Optional[np.ndarray] = None

    # ---------- factory ----------
    @classmethod
    def from_image_sets(
        cls,
        pretrained_model_name: str,
        image_paths_set1: List,
        image_paths_set2: Optional[List] = None,
        batch_size: int = 5,
        metric: str = "cosine",
        strategy: str = "mean_patches",
    ):
        if image_paths_set2 is None:
            image_paths_set2 = image_paths_set1
        
        model, processor = load_vision_model(pretrained_model_name)

        store1 = ImageFeatureStore.from_images_with_model(
            image_paths=image_paths_set1,
            model=model,
            processor=processor,
            model_name=pretrained_model_name,
            batch_size=batch_size,
            strategy=strategy,
        )

        store2 = ImageFeatureStore.from_images_with_model(
            image_paths=image_paths_set2,
            model=model,
            processor=processor,
            model_name=pretrained_model_name,
            batch_size=batch_size,
            strategy=strategy,
        )

        return cls(store1, store2, metric=metric)

    # ---------- core ----------
    def compute(self) -> np.ndarray:
        if self._similarity_matrix is not None:
            return self._similarity_matrix

        if self.metric == "cosine":
            sim = cosine_similarity_matrix(
                self.store1.features,
                self.store2.features,
            )
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        self._similarity_matrix = sim
        return sim

    @property
    def matrix(self) -> np.ndarray:
        return self.compute()

    # ---------- plotting ----------
    def plot(self, **plot_kwargs) -> np.ndarray:
        from .plot_similarity_matrix import plot_matrix

        if not hasattr(self.store1, "image_paths") or not hasattr(self.store2, "image_paths"):
            raise AttributeError(
                "Both stores must have an `image_paths` attribute to use plot()."
            )

        def _load(p):
            if isinstance(p, Image.Image):
                return p
            with Image.open(p) as im:
                return im.convert("RGB").copy()

        images1 = [_load(p) for p in self.store1.image_paths]
        images2 = [_load(p) for p in self.store2.image_paths]

        return plot_matrix(images1, images2, self.matrix, **plot_kwargs)

    # Dunder methods
    def __str__(self) -> str:
        return f"FeatureSimilarity(store1={self.store1}, store2={self.store2}, metric={self.metric})"

    def __repr__(self) -> str:
        return f"FeatureSimilarity(store1={self.store1}, store2={self.store2}, metric={self.metric})"

    def __len__(self) -> int:
        return len(self.store1)
    
    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        return self.store1[idx], self.store2[idx]

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other) -> bool:
        if not isinstance(other, FeatureSimilarity):
            return False
        return self.store1 == other.store1 and self.store2 == other.store2 and self.metric == other.metric