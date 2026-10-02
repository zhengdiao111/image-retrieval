from .database import get_table

from .embeddings import (
    ImageEmbedder,
)


class ImageSearcher:

    def __init__(self):

        self.table = (
            get_table()
        )

        self.embedder = (
            ImageEmbedder()
        )


    def search(
        self,
        query: str,
        limit: int = 20,
    ):

        query_vector = (
            self.embedder
            .embed_text(
                query
            )
        )

        results = (
            self.table
            .search(
                query_vector
            )
            .distance_type(
                "cosine"
            )
            .select(
                [
                    "path",
                    "filename",
                    "thumbnail_path",
                    "date_taken",
                    "latitude",
                    "longitude",
                    "description",
                    "camera_model",
                ]
            )
            .limit(
                limit
            )
            .to_pandas()
        )

        if "_distance" in results:

            results[
                "similarity"
            ] = (
                1.0
                - results[
                    "_distance"
                ]
            )

        return results