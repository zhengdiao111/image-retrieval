# Image Retrieval

A local-first image retrieval system for fast searching across large personal photo collections using both:

- **Image content** — semantic search such as `"dog in snow"`, `"sunset over a lake"`, or `"people standing outdoors"`
- **Image metadata** — date taken, GPS/location, camera information, filenames, and Google Photos metadata

The project precomputes image embeddings once and stores them in a local vector database so searches do not need to re-analyze every image.

The system is designed to work with local folders today and later with image libraries stored on a NAS.

---

## Overview

The indexing pipeline is:

```text
Image folder
    │
    ├── JPG / JPEG / PNG / WEBP / HEIC
    │
    └── Google supplemental metadata JSON
              │
              ▼
        Incremental indexer
              │
      ┌───────┴────────┐
      ▼                ▼
Metadata extraction   SigLIP2
                      image embedding
      │                │
      └───────┬────────┘
              ▼
           LanceDB
              │
              ▼
        Semantic search
```

Once an image has been indexed, future searches operate on the stored vector and metadata rather than reopening and analyzing the original image.

---

# Features

Current and planned features include:

- Local semantic image retrieval
- SigLIP2 image/text embeddings
- GPU acceleration through PyTorch/CUDA
- LanceDB vector storage
- Google Photos supplemental metadata support
- EXIF metadata extraction
- GPS metadata extraction
- Capture-date extraction
- Camera make/model extraction
- Incremental indexing
- Automatic skipping of unchanged files
- Batch GPU embedding
- Thumbnail generation
- HEIC/HEIF support
- Text-to-image search
- Future metadata filters
- Future NAS support
- Future automatic folder monitoring
- Future Streamlit browser interface
- Future image-to-image search
- Future document retrieval integration

---

# Technology Stack

The current implementation uses:

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package manager | `uv` |
| Image-text model | `google/siglip2-base-patch16-224` |
| Deep learning | PyTorch |
| Model interface | Hugging Face Transformers |
| Vector database | LanceDB |
| Data format | Apache Arrow |
| Image processing | Pillow |
| HEIC support | pillow-heif |
| Metadata | EXIF + Google Photos JSON |
| Future folder monitoring | Watchdog |
| Future UI | Streamlit |

---

# Project Structure

```text
image-retrieval/
│
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
│
├── data/
│   ├── image_db/
│   └── thumbnails/
│
├── src/
│   └── image_retrieval/
│       ├── __init__.py
│       ├── config.py
│       ├── metadata.py
│       ├── embeddings.py
│       ├── database.py
│       ├── indexer.py
│       └── search.py
│
├── scripts/
│   ├── test_metadata.py
│   ├── test_model.py
│   ├── test_database.py
│   ├── index_images.py
│   └── search_images.py
│
└── tests/
```

---

# Installation

## 1. Clone or create the project

```powershell
git clone <repository-url>
cd image-retrieval
```

Or for a new local copy:

```powershell
mkdir image-retrieval
cd image-retrieval

uv init --python 3.12
git init
```

---

## 2. Install dependencies

```powershell
uv add transformers accelerate
uv add pillow pillow-heif
uv add lancedb pyarrow
uv add python-dotenv
uv add pandas
uv add watchdog
uv add streamlit
```

Install PyTorch with GPU support:

```powershell
uv pip install torch torchvision --torch-backend=auto
```

Check CUDA:

```powershell
uv run python
```

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

A CUDA-enabled system should report:

```text
CUDA available: True
```

---

# Configuration

Create a `.env` file in the project root:

```env
IMAGE_ROOT=C:/Users/USERNAME/Photos

DB_PATH=./data/image_db
THUMBNAIL_PATH=./data/thumbnails

EMBEDDING_MODEL=google/siglip2-base-patch16-224

BATCH_SIZE=16
```

On Windows, forward slashes are recommended:

```text
C:/Users/USERNAME/Photos
```

rather than:

```text
C:\Users\USERNAME\Photos
```

---

# Supported Image Formats

The current configuration supports:

```text
.jpg
.jpeg
.png
.webp
.heic
.heif
```

Additional formats can be added in `config.py`.

---

# Google Photos Export Support

Google Photos exports commonly contain an image and a corresponding supplemental metadata file.

Example:

```text
IMG_20240311_130002283.jpg

IMG_20240311_130002283.jpg.supplemental-metadata.json
```

These files should remain in the same directory.

The indexer processes the image and automatically searches for the matching supplemental metadata file.

For example:

```text
IMG_20240311_130002283.jpg
        │
        ├── image pixels
        ├── EXIF metadata
        │
        └── matching sidecar
              ↓
IMG_20240311_130002283.jpg.supplemental-metadata.json
```

The Google sidecar may provide:

- Photo capture time
- GPS coordinates
- Google Photos description
- Original title
- Additional metadata not preserved in the downloaded image EXIF

---

# Metadata Priority

Different metadata sources are used for different fields.

## Capture date

Priority:

```text
Google photoTakenTime
        ↓
EXIF DateTimeOriginal
        ↓
filesystem timestamp
```

## GPS location

Priority:

```text
Google geoData
        ↓
Google geoDataExif
        ↓
EXIF GPS
```

## Camera information

Camera make and model are read from image EXIF metadata.

## Image dimensions

Width and height are read directly from the image file.

## Description

Google Photos descriptions are read from the supplemental metadata JSON when available.

---

# Testing Metadata Extraction

Run:

```powershell
uv run python scripts/test_metadata.py
```

Example output:

```text
{
    'camera_make': 'motorola',
    'camera_model': 'motorola edge 2023',
    'date_taken': '2024-03-11T...',
    'date_source': 'google_sidecar',
    'latitude': ...,
    'longitude': ...,
    'filename': 'IMG_20240311_130002283.jpg',
    'sidecar_path':
        '...IMG_20240311_130002283.jpg.supplemental-metadata.json'
}
```

If:

```text
sidecar_path = None
```

the supplemental metadata file was not found.

---

# Image Embeddings

Semantic retrieval uses:

```text
google/siglip2-base-patch16-224
```

Images and text are encoded into the same embedding space.

Example:

```text
image
  ↓
SigLIP2
  ↓
768-dimensional vector
```

and:

```text
"dog running through snow"
          ↓
       SigLIP2
          ↓
768-dimensional vector
```

The vectors can then be compared using cosine similarity.

---

# Testing the Embedding Model

Run:

```powershell
uv run python scripts/test_model.py
```

Expected output:

```text
Embedding device: cuda

Image: (768,)
Text: (768,)
```

The first run downloads the SigLIP2 model.

---

# LanceDB Storage

Each indexed image is stored as one LanceDB record.

Example fields:

```text
id

path
filename

file_size
modified_ns

sidecar_path
sidecar_size
sidecar_modified_ns

width
height

camera_make
camera_model

date_taken
date_source

latitude
longitude

description
google_title

embedding_model
embedding_version

thumbnail_path

vector
```

The vector is a 768-dimensional normalized SigLIP2 embedding.

---

# Incremental Indexing

The project is designed so images are **not embedded repeatedly**.

Run:

```powershell
uv run python scripts/index_images.py
```

On the first run:

```text
Images found: 274
Already current: 0
Need indexing: 274
```

The indexer:

1. Finds supported image files
2. Checks Google supplemental metadata
3. Extracts metadata
4. Loads the image
5. Creates a SigLIP2 embedding
6. Creates a thumbnail
7. Stores everything in LanceDB

A second run should produce:

```text
Images found: 274
Already current: 274
Need indexing: 0

Nothing to do.
```

This means no model loading and no GPU inference are required when nothing changed.

---

# Change Detection

The indexer currently checks:

```text
image file size
image modified timestamp

sidecar file size
sidecar modified timestamp
```

If these match the stored database values, the image is skipped.

Conceptually:

```text
image found
    │
    ▼
already indexed?
    │
 ┌──┴──┐
 NO    YES
 │      │
 ▼      ▼
INDEX  changed?
        │
     ┌──┴──┐
    YES    NO
     │      │
     ▼      ▼
 REINDEX   SKIP
```

Nanosecond timestamps are loaded directly through Apache Arrow instead of Pandas to avoid precision loss when comparing large `int64` timestamp values.

---

# Batch Processing

Images are embedded in batches for improved GPU performance.

The default is:

```env
BATCH_SIZE=16
```

For example:

```text
16 images
    ↓
SigLIP2 GPU inference
    ↓
16 embeddings
```

Larger batch sizes such as 32 or 64 may be faster if GPU memory allows.

---

# Thumbnail Generation

Each indexed image gets a smaller thumbnail.

Example:

```text
data/thumbnails/
├── 2489a5c....jpg
├── 74a9d1f....jpg
└── ...
```

Thumbnails are used for future search interfaces so the original high-resolution image does not need to be loaded just to display results.

---

# Semantic Search

Once indexing is complete, search with:

```powershell
uv run python scripts/search_images.py "dog playing in snow"
```

Other examples:

```powershell
uv run python scripts/search_images.py "sunset over a lake"
```

```powershell
uv run python scripts/search_images.py "people standing outdoors"
```

```powershell
uv run python scripts/search_images.py "car parked beside a building"
```

The search process is:

```text
text query
    ↓
SigLIP2 text embedding
    ↓
LanceDB cosine similarity search
    ↓
top matching images
```

The original photo collection is not scanned or re-embedded during search.

---

# Retrieval Architecture

The project separates expensive indexing from interactive retrieval.

## Indexing

```text
Original images
       ↓
metadata extraction
       +
SigLIP2 embedding
       +
thumbnail generation
       ↓
LanceDB
```

## Search

```text
query
  ↓
text embedding
  ↓
LanceDB
  ↓
matching image records
```

This design allows very fast repeated searches.

---

# Future Metadata Filtering

Planned filters include:

- Date range
- GPS/location
- Camera
- Image dimensions
- Whether GPS information exists
- Google Photos description
- File type

Example future query:

```text
"waterfall"

AND

date between:
2025-09-01 and 2025-11-30

AND

location:
Pennsylvania
```

Semantic vector search can then be combined with exact metadata constraints.

---

# Future NAS Integration

The project is designed to eventually work with a locally hosted NAS.

Example layout:

```text
NAS
│
├── Photos/
├── Documents/
├── Research/
├── Backups/
└── AI-Projects/
```

The NAS stores original files while the PC performs GPU-heavy AI processing.

Recommended architecture:

```text
NAS
│
│ SMB / Ethernet
▼
Windows GPU PC
│
├── SigLIP2
├── LanceDB
├── thumbnails
└── retrieval application
```

The live LanceDB database should preferably remain on the PC's NVMe SSD for low-latency access.

The NAS should hold the original image files.

Example future `.env`:

```env
IMAGE_ROOT=//NASNAME/Photos

DB_PATH=./data/image_db
THUMBNAIL_PATH=./data/thumbnails
```

A UNC path is preferred over a mapped Windows drive letter.

---

# Future Automatic Folder Monitoring

The project includes `watchdog` as a dependency for future automatic ingestion.

Planned workflow:

```text
new photo arrives
       ↓
watch_images.py detects it
       ↓
wait for file copy to finish
       ↓
extract metadata
       ↓
generate embedding
       ↓
generate thumbnail
       ↓
update LanceDB
```

The regular:

```powershell
uv run python scripts/index_images.py
```

command will remain the authoritative reconciliation process.

---

# Future User Interface

A Streamlit interface is planned for interactive browsing.

Example:

```text
Search

[ golden retriever in snow          ]

Date
[ 2024-01-01 ] → [ 2026-12-31 ]

Location
[ New Jersey                      ]

              Search


┌────────┐ ┌────────┐ ┌────────┐
│ photo  │ │ photo  │ │ photo  │
└────────┘ └────────┘ └────────┘
```

The interface will use thumbnails for fast display and open the original file only when requested.

---

# Future Document Retrieval

The long-term project may extend beyond images.

A NAS can contain both:

```text
Photos/
Documents/
Research/
```

Image files can use:

```text
SigLIP2
```

while documents can use:

```text
PDF/DOCX/PPTX
      ↓
text extraction
      ↓
chunking
      ↓
text embeddings
      ↓
vector retrieval
```

This would eventually allow one local search system to retrieve both images and documents.

---

# Git

The following should **not** be committed:

```text
.env

.venv/

data/image_db/
data/thumbnails/

photos
embeddings
database files
```

The repository should contain:

```text
source code
tests
README
configuration templates
project metadata
```

---

# Current Development Status

Working:

- [x] Python/`uv` environment
- [x] CUDA/PyTorch
- [x] SigLIP2 image embeddings
- [x] SigLIP2 text embeddings
- [x] Google supplemental metadata matching
- [x] EXIF metadata extraction
- [x] LanceDB table creation
- [x] Batch image indexing
- [x] Incremental indexing
- [x] Thumbnail generation
- [x] Semantic text-to-image retrieval

Planned:

- [ ] Date filtering
- [ ] Location filtering
- [ ] EXIF GPS fallback
- [ ] Approximate nearest-neighbor vector index
- [ ] Automatic folder watcher
- [ ] Streamlit search interface
- [ ] Image-to-image retrieval
- [ ] Natural-language metadata query parsing
- [ ] NAS migration
- [ ] Phone-accessible search interface
- [ ] Document retrieval integration

---

# Example Workflow

Initial indexing:

```powershell
uv run python scripts/index_images.py
```

Verify no unnecessary re-indexing:

```powershell
uv run python scripts/index_images.py
```

Expected:

```text
Already current: 274
Need indexing: 0
```

Search:

```powershell
uv run python scripts/search_images.py "mountains at sunset"
```

Add new images to the photo directory and run:

```powershell
uv run python scripts/index_images.py
```

Only new or changed images should be processed.

---

# Design Principles

This project follows several core principles:

1. **Index once, retrieve many times**
2. **Do not re-analyze unchanged images**
3. **Use deterministic metadata filters for dates and locations**
4. **Use embeddings for semantic visual meaning**
5. **Keep original files separate from the vector database**
6. **Keep expensive AI work on the GPU-equipped PC**
7. **Keep the storage backend replaceable**
8. **Allow migration from local folders to NAS storage without redesigning the search system**

---
