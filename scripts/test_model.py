from PIL import Image

from image_retrieval.embeddings import (
    ImageEmbedder,
)


image_path = (
    "C:/Users/Zheng_Diao/Downloads/PHOTOS/"
    "IMG_20240311_130002283.jpg"
)

embedder = ImageEmbedder()

image = Image.open(
    image_path
).convert("RGB")

image_vector = (
    embedder.embed_images(
        [image]
    )[0]
)

text_vector = (
    embedder.embed_text(
        "outdoor photograph"
    )
)

print(
    "Image:",
    image_vector.shape,
)

print(
    "Text:",
    text_vector.shape,
)