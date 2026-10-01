import json

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ExifTags

from .config import GOOGLE_SIDECAR_SUFFIX


def find_google_sidecar(
    image_path: Path,
) -> Path | None:

    candidates = [
        image_path.with_name(
            image_path.name + GOOGLE_SIDECAR_SUFFIX
        ),

        # Also support older/alternate Takeout naming
        image_path.with_name(
            image_path.name + ".json"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_google_metadata(
    image_path: Path,
) -> dict | None:

    sidecar = find_google_sidecar(image_path)

    if sidecar is None:
        return None

    try:
        with open(
            sidecar,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        print(
            f"Warning: could not read "
            f"{sidecar}: {exc}"
        )

        return None


def google_time_to_iso(
    data: dict | None,
) -> str | None:

    if not data:
        return None

    timestamp = data.get("timestamp")

    if timestamp is not None:
        try:
            return datetime.fromtimestamp(
                int(timestamp),
                tz=timezone.utc,
            ).isoformat()
        except (ValueError, TypeError):
            pass

    return data.get("formatted")


def valid_coordinates(
    latitude,
    longitude,
):

    if latitude is None or longitude is None:
        return False

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (ValueError, TypeError):
        return False

    # Google frequently uses 0,0 when location
    # information is unavailable.
    if latitude == 0 and longitude == 0:
        return False

    if not -90 <= latitude <= 90:
        return False

    if not -180 <= longitude <= 180:
        return False

    return True


def extract_metadata(
    path: Path,
) -> dict:

    path = path.resolve()

    stat = path.stat()

    result = {
        "path": str(path),
        "filename": path.name,

        "file_size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,

        "width": None,
        "height": None,

        "camera_make": None,
        "camera_model": None,

        "date_taken": None,
        "date_source": None,

        "latitude": None,
        "longitude": None,

        "description": None,
        "google_title": None,

        "sidecar_path": None,
        "sidecar_size": None,
        "sidecar_modified_ns": None,
    }

    # --------------------------------------------------
    # Image / EXIF
    # --------------------------------------------------

    with Image.open(path) as img:

        result["width"] = img.width
        result["height"] = img.height

        exif = img.getexif()

        if exif:

            result["camera_make"] = exif.get(
                ExifTags.Base.Make
            )

            result["camera_model"] = exif.get(
                ExifTags.Base.Model
            )

            exif_date = exif.get(
                ExifTags.Base.DateTimeOriginal
            )

            if exif_date:
                result["date_taken"] = str(
                    exif_date
                )

                result["date_source"] = "exif"

    # --------------------------------------------------
    # Google supplemental metadata
    # --------------------------------------------------

    sidecar = find_google_sidecar(path)

    if sidecar is not None:

        sidecar_stat = sidecar.stat()

        result["sidecar_path"] = str(
            sidecar.resolve()
        )

        result["sidecar_size"] = (
            sidecar_stat.st_size
        )

        result["sidecar_modified_ns"] = (
            sidecar_stat.st_mtime_ns
        )

        google = load_google_metadata(path)

        if google is not None:

            result["description"] = (
                google.get("description")
            )

            result["google_title"] = (
                google.get("title")
            )

            # ------------------------------------------
            # Capture time
            # ------------------------------------------

            google_time = google_time_to_iso(
                google.get("photoTakenTime")
            )

            if google_time:

                result["date_taken"] = (
                    google_time
                )

                result["date_source"] = (
                    "google_sidecar"
                )

            # ------------------------------------------
            # GPS
            # ------------------------------------------

            geo = google.get(
                "geoData",
                {}
            )

            latitude = geo.get(
                "latitude"
            )

            longitude = geo.get(
                "longitude"
            )

            if valid_coordinates(
                latitude,
                longitude,
            ):

                result["latitude"] = float(
                    latitude
                )

                result["longitude"] = float(
                    longitude
                )

            # Some Takeout files put useful
            # coordinates under geoDataExif.
            if (
                result["latitude"] is None
                or result["longitude"] is None
            ):

                geo_exif = google.get(
                    "geoDataExif",
                    {}
                )

                latitude = geo_exif.get(
                    "latitude"
                )

                longitude = geo_exif.get(
                    "longitude"
                )

                if valid_coordinates(
                    latitude,
                    longitude,
                ):

                    result["latitude"] = float(
                        latitude
                    )

                    result["longitude"] = float(
                        longitude
                    )

    return result