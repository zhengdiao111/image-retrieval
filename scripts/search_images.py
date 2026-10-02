import argparse

from image_retrieval.search import (
    ImageSearcher,
)


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    searcher = (
        ImageSearcher()
    )

    results = (
        searcher.search(
            query=args.query,
            limit=args.limit,
        )
    )

    if len(results) == 0:

        print(
            "\nNo results."
        )

        return

    print()

    for index, row in (
        results.iterrows()
    ):

        print(
            f"{index + 1}. "
            f"{row['filename']}"
        )

        print(
            f"   similarity: "
            f"{row['similarity']:.3f}"
        )

        print(
            f"   {row['path']}"
        )

        if row[
            "date_taken"
        ]:

            print(
                f"   date: "
                f"{row['date_taken']}"
            )

        print()


if __name__ == "__main__":

    main()