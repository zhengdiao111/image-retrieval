import hashlib
from pathlib import Path

from PIL import (
    Image,
    ImageOps,
)

from pillow_heif import (
    register_heif_opener,
)

from .config import (
    IMAGE_ROOT,
    THUMBNAIL_PATH,
    SUPPORTED_EXTENSIONS,
    BATCH_SIZE,
    EMBEDDING_MODEL,
)

from .database import get_table

from .embeddings import ImageEmbedder

from .metadata import (
    extract_metadata,
    find_google_sidecar,
)


register_heif_opener()


EMBEDDING_VERSION = 1

THUMBNAIL_SIZE = (
    512,
    512,
)


def make_image_id(
    path: Path,
) -> str:

    canonical = str(
        path.resolve()
    ).casefold()

    return hashlib.sha1(
        canonical.encode("utf-8")
    ).hexdigest()


def iter_images():

    root = Path(
        IMAGE_ROOT
    )

    if not root.exists():

        raise FileNotFoundError(
            f"IMAGE_ROOT does not exist: "
            f"{root}"
        )

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            yield path


def get_file_signature(
    path: Path,
):

    stat = path.stat()

    sidecar = find_google_sidecar(
        path
    )

    sidecar_size = None
    sidecar_modified_ns = None

    if sidecar is not None:

        sidecar_stat = (
            sidecar.stat()
        )

        sidecar_size = (
            sidecar_stat.st_size
        )

        sidecar_modified_ns = (
            sidecar_stat.st_mtime_ns
        )

    return {
        "file_size":
            stat.st_size,

        "modified_ns":
            stat.st_mtime_ns,

        "sidecar_size":
            sidecar_size,

        "sidecar_modified_ns":
            sidecar_modified_ns,
    }


def load_existing_state(
    table,
):

    columns = [
        "id",
        "path",
        "file_size",
        "modified_ns",
        "sidecar_size",
        "sidecar_modified_ns",
    ]

    # Use Arrow instead of Pandas.
    # This preserves int64 nanosecond timestamps
    # exactly, even when the column also
    # contains null values.
    rows = (
        table
        .search()
        .select(columns)
        .to_arrow()
        .to_pylist()
    )

    existing = {}

    for row in rows:

        existing[row["id"]] = {
            "path":
                row["path"],

            "file_size":
                row["file_size"],

            "modified_ns":
                row["modified_ns"],

            "sidecar_size":
                row["sidecar_size"],

            "sidecar_modified_ns":
                row[
                    "sidecar_modified_ns"
                ],
        }

    return existing


def needs_reindex(
    existing_record,
    current_signature,
):

    if existing_record is None:
        return True

    fields = [
        "file_size",
        "modified_ns",
        "sidecar_size",
        "sidecar_modified_ns",
    ]

    for field in fields:

        old = existing_record.get(
            field
        )

        new = current_signature.get(
            field
        )

        # Normalize integer values while
        # preserving None.
        if old is not None:
            old = int(old)

        if new is not None:
            new = int(new)

        if old != new:
            return True

    return False


def create_thumbnail(
    image: Image.Image,
    image_id: str,
) -> str:

    thumbnail_root = Path(
        THUMBNAIL_PATH
    )

    thumbnail_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        thumbnail_root
        / f"{image_id}.jpg"
    )

    thumbnail = image.copy()

    thumbnail.thumbnail(
        THUMBNAIL_SIZE,
        Image.Resampling.LANCZOS,
    )

    thumbnail.save(
        output_path,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    return str(
        output_path.resolve()
    )


def process_batch(
    paths,
    embedder,
):

    images = []
    metadata_list = []
    image_ids = []
    valid_paths = []

    for path in paths:

        try:

            metadata = (
                extract_metadata(
                    path
                )
            )

            image_id = (
                make_image_id(
                    path
                )
            )

            with Image.open(
                path
            ) as img:

                img = (
                    ImageOps
                    .exif_transpose(
                        img
                    )
                )

                img = (
                    img
                    .convert("RGB")
                )

                img.load()

                image = (
                    img.copy()
                )

            images.append(
                image
            )

            metadata_list.append(
                metadata
            )

            image_ids.append(
                image_id
            )

            valid_paths.append(
                path
            )

        except Exception as exc:

            print(
                f"\nERROR reading "
                f"{path}"
            )

            print(
                f"    {exc}"
            )

    if not images:
        return []

    vectors = (
        embedder
        .embed_images(
            images
        )
    )

    records = []

    for (
        path,
        image,
        metadata,
        image_id,
        vector,
    ) in zip(
        valid_paths,
        images,
        metadata_list,
        image_ids,
        vectors,
    ):

        thumbnail_path = (
            create_thumbnail(
                image,
                image_id,
            )
        )

        record = {
            "id":
                image_id,

            **metadata,

            "embedding_model":
                EMBEDDING_MODEL,

            "embedding_version":
                EMBEDDING_VERSION,

            "thumbnail_path":
                thumbnail_path,

            "vector":
                vector
                .astype("float32")
                .tolist(),
        }

        records.append(
            record
        )

    return records


def upsert_records(
    table,
    records,
):

    if not records:
        return

    (
        table
        .merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(records)
    )


def run_index():

    print(
        "\nImage Retrieval Indexer"
    )

    print(
        f"\nImage root:\n"
        f"{IMAGE_ROOT}"
    )

    table = get_table()

    existing = (
        load_existing_state(
            table
        )
    )

    image_paths = list(
        iter_images()
    )

    print(
        f"\nImages found: "
        f"{len(image_paths)}"
    )

    pending = []

    skipped = 0

    for path in image_paths:

        image_id = (
            make_image_id(
                path
            )
        )

        signature = (
            get_file_signature(
                path
            )
        )

        old = existing.get(
            image_id
        )

        if needs_reindex(
            old,
            signature,
        ):

            pending.append(
                path
            )

        else:

            skipped += 1

    print(
        f"Already current: "
        f"{skipped}"
    )

    print(
        f"Need indexing: "
        f"{len(pending)}"
    )

    if not pending:

        print(
            "\nNothing to do."
        )

        return

    embedder = (
        ImageEmbedder()
    )

    total_batches = (
        len(pending)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    indexed = 0

    for batch_number, start in enumerate(
        range(
            0,
            len(pending),
            BATCH_SIZE,
        ),
        start=1,
    ):

        batch_paths = pending[
            start:
            start + BATCH_SIZE
        ]

        print(
            f"\nBatch "
            f"{batch_number}/"
            f"{total_batches}"
        )

        records = (
            process_batch(
                batch_paths,
                embedder,
            )
        )

        upsert_records(
            table,
            records,
        )

        indexed += len(
            records
        )

        print(
            f"Indexed: "
            f"{indexed}/"
            f"{len(pending)}"
        )

    print(
        "\nFinished."
    )

    print(
        f"Indexed/updated: "
        f"{indexed}"
    )

    print(
        f"Skipped: "
        f"{skipped}"
    )