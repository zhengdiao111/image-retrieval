import torch
import torch.nn.functional as F

from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
)

from .config import EMBEDDING_MODEL


class ImageEmbedder:

    def __init__(self):

        # Use NVIDIA GPU when available.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Embedding device: {self.device}"
        )

        # Load the processor used for both
        # image and text inputs.
        self.processor = (
            AutoProcessor.from_pretrained(
                EMBEDDING_MODEL
            )
        )

        # Load the SigLIP / SigLIP2 model.
        self.model = (
            AutoModel.from_pretrained(
                EMBEDDING_MODEL
            )
        )

        # Move model to GPU or CPU.
        self.model.to(self.device)

        # Disable training-specific behavior.
        self.model.eval()


    def embed_images(
        self,
        images: list[Image.Image],
    ):
        """
        Convert one or more PIL images into
        normalized embedding vectors.

        Returns:
            NumPy array with shape:
            (number_of_images, embedding_dimension)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        # Move processor outputs to the same
        # device as the model.
        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        # Generate image embeddings.
        with torch.inference_mode():

            outputs = (
                self.model
                .get_image_features(
                    **inputs
                )
            )

            # Transformers 5.x returns
            # BaseModelOutputWithPooling.
            # The actual embedding tensor is
            # stored in pooler_output.
            features = (
                outputs.pooler_output
            )

            # Normalize each embedding to
            # unit length.
            features = F.normalize(
                features,
                p=2,
                dim=-1,
            )

        return (
            features
            .cpu()
            .numpy()
        )


    def embed_text(
        self,
        text: str,
    ):
        """
        Convert a text query into a normalized
        embedding vector.

        Returns:
            NumPy array with shape:
            (embedding_dimension,)
        """

        inputs = self.processor(
            text=[text],
            padding="max_length",
            return_tensors="pt",
        )

        # Move processor outputs to the same
        # device as the model.
        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        # Generate text embedding.
        with torch.inference_mode():

            outputs = (
                self.model
                .get_text_features(
                    **inputs
                )
            )

            # Transformers 5.x returns
            # BaseModelOutputWithPooling.
            features = (
                outputs.pooler_output
            )

            # Normalize so image and text
            # embeddings can be compared
            # using cosine similarity /
            # dot product.
            features = F.normalize(
                features,
                p=2,
                dim=-1,
            )

        # Only one text query was provided,
        # so return the first embedding.
        return (
            features[0]
            .cpu()
            .numpy()
        )