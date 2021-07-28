"""
create_download_tables.py

Build and verify download link table.
"""

import requests


def verify_download_link(link):
    results = requests.head(link)
    return results.ok


def produce_download_tables():
    sizes = ["medium", "small"]

    runs = {
        "small": ["Alias", "Battlestar", "Caprica", "Darkmatter", "Expanse"],
        "medium": ["Arwen", "Beren", "Cerebrimbor", "Durin", "Eowyn"],
    }

    run_to_id = {
        "Alias": "x21",
        "Battlestar": "x49",
        "Caprica": "x81",
        "Darkmatter": "x343",
        "Expanse": "x777",
        "Arwen": "x21",
        "Beren": "x49",
        "Cerebrimbor": "x81",
        "Durin": "x343",
        "Eowyn": "x777",
    }

    checkpoints = [100000, 200000, 400000]

    download_sizes = {"small": "1.8G", "medium": "4.9G"}

    tables = []
    for size in sizes:
        table = (
            f".. csv-table:: GPT-2 {size.capitalize()} Models\n"
            '   :header: "Run", "Type", "Checkpoint", "Size", "Link"\n'
            "   :widths: 7,7,7,5,7\n\n"
        )
        for run in sorted(runs[size]):
            for checkpoint in sorted(checkpoints, reverse=True):
                # build and verify download link
                download_link = f"https://storage.googleapis.com/mistral-models/gpt2-{size}/{run.lower()}-gpt2-{size}-{run_to_id[run]}/{run.lower()}-checkpoint-{checkpoint}.zip"
                # assert verify_download_link(download_link), f"link failed: {download_link}"
                # add row
                table += (
                    f'   "{run}", "GPT-2 {size.capitalize()}", "{checkpoint}", {download_sizes[size]}, `download'
                    f" <{download_link}>`_\n"
                )
        tables.append(table)

    return tables


if __name__ == "__main__":
    print("")
    for table in produce_download_tables():
        print(table)
    print("")
