import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = Path(
    os.getenv("IMAGE_ROOT", "")
).expanduser()

DB_PATH = Path(
    os.getenv(
        "DB_PATH",
        str(PROJECT_ROOT / "data" / "image_db"),
    )
)

THUMBNAIL_PATH = Path(
    os.getenv(
        "THUMBNAIL_PATH",
        str(PROJECT_ROOT / "data" / "thumbnails"),
    )
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "google/siglip2-base-patch16-224",
)

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "16")
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
}


GOOGLE_SIDECAR_SUFFIX = ".supplemental-metadata"