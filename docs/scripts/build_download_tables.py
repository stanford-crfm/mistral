"""
create_download_tables.py

Build and verify download link table.
"""

import argparse

import requests


def verify_download_link(link):
    results = requests.head(link)
    return results.ok


def github_table_header(size):
    return (
        f"\nGPT-2 {size.capitalize()}\n\n| Run | Type | Checkpoint | Size | Link |\n| --- | --- | --- | --- | --- |\n"
    )


def github_table_row(run, size, checkpoint, download_size, download_link):
    return f"| {run} | GPT-2 {size.capitalize()} | {checkpoint} | {download_size} | [download]({download_link}) |\n"


def rst_table_header(size):
    return (
        f".. csv-table:: GPT-2 {size.capitalize()} Models\n"
        '   :header: "Run", "Type", "Checkpoint", "Size", "Link"\n'
        "   :widths: 7,7,7,5,7\n\n"
    )


def rst_table_row(run, size, checkpoint, download_size, download_link):
    return f'   "{run}", "GPT-2 {size.capitalize()}", "{checkpoint}", {download_size}, `download <{download_link}>`_\n'


table_header_creators = {"github": github_table_header, "rst": rst_table_header}
row_creators = {"github": github_table_row, "rst": rst_table_row}


def produce_download_tables(mode="rst"):
    sizes = ["medium", "small"]

    runs = {
        "small": ["Alias", "Battlestar", "Caprica", "Darkmatter", "Expanse"],
        "medium": ["Arwen", "Beren", "Celebrimbor", "Durin", "Eowyn"],
    }

    run_to_seed = {
        "Alias": "x21",
        "Battlestar": "x49",
        "Caprica": "x81",
        "Darkmatter": "x343",
        "Expanse": "x777",
        "Arwen": "x21",
        "Beren": "x49",
        "Celebrimbor": "x81",
        "Durin": "x343",
        "Eowyn": "x777",
    }

    checkpoints = [100000, 200000, 300000, 400000]

    download_sizes = {"small": "1.8G", "medium": "4.9G"}

    tables = []
    for size in sizes:
        table = table_header_creators[mode](size)
        for run in sorted(runs[size]):
            for checkpoint in sorted(checkpoints, reverse=True):
                # build and verify download link
                download_link = f"https://storage.googleapis.com/mistral-models/gpt2-{size}/{run.lower()}-gpt2-{size}-{run_to_seed[run]}/{run.lower()}-{run_to_seed[run]}-checkpoint-{checkpoint}.zip"
                # assert verify_download_link(download_link), f"link failed: {download_link}"
                # add row
                table += row_creators[mode](run, size, checkpoint, download_sizes[size], download_link)
        tables.append(table)

    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["github", "rst"], help="type of table to build", default="rst")
    args = parser.parse_args()
    print("")
    for table in produce_download_tables(mode=args.mode):
        print(table)
    print("")
