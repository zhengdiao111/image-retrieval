from image_retrieval.database import (
    get_table,
)


table = get_table()

print(table.schema)

print(
    "Rows:",
    table.count_rows(),
)