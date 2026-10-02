from pathlib import Path

import lancedb
import pyarrow as pa

from .config import DB_PATH


TABLE_NAME = "images"

VECTOR_DIM = 768


IMAGE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),

        pa.field("path", pa.string()),
        pa.field("filename", pa.string()),

        pa.field("file_size", pa.int64()),
        pa.field("modified_ns", pa.int64()),

        pa.field("sidecar_path", pa.string()),
        pa.field("sidecar_size", pa.int64()),
        pa.field(
            "sidecar_modified_ns",
            pa.int64(),
        ),

        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),

        pa.field("camera_make", pa.string()),
        pa.field("camera_model", pa.string()),

        pa.field("date_taken", pa.string()),
        pa.field("date_source", pa.string()),

        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),

        pa.field("description", pa.string()),
        pa.field("google_title", pa.string()),

        pa.field(
            "embedding_model",
            pa.string(),
        ),

        pa.field(
            "embedding_version",
            pa.int32(),
        ),

        pa.field(
            "thumbnail_path",
            pa.string(),
        ),

        pa.field(
            "vector",
            pa.list_(
                pa.float32(),
                VECTOR_DIM,
            ),
        ),
    ]
)


def get_database():

    db_path = Path(DB_PATH)

    db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return lancedb.connect(
        str(db_path)
    )


def get_table():

    db = get_database()

    table = db.create_table(
        TABLE_NAME,
        schema=IMAGE_SCHEMA,
        exist_ok=True,
    )

    return table