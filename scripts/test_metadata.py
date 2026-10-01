from pathlib import Path
from pprint import pprint

from image_retrieval.metadata import (
    extract_metadata,
)


image = Path(
    "C:/Users/Zheng_Diao/Downloads/PHOTOS/"
    "IMG_20240311_130002283.jpg"
)

metadata = extract_metadata(image)

pprint(metadata)