import json
import sys

prefix = " ".join([f"<|prefix{x}|>" for x in range(1,10)])

examples = json.loads(open(sys.argv[1]).read())

split_to_id_lists = json.loads(open("pubmed_splits.json").read())
id_to_split = {}
for split in split_to_id_lists:
    for example_id in split_to_id_lists[split]:
        id_to_split[example_id] = split


def biogpt_version(example):
    context = " ".join(example["CONTEXTS"])
    question = example["QUESTION"]
    answer = example["LONG_ANSWER"]
    label = example["final_decision"]
    source = f"question: {question} context: {context} answer: {answer}"
    target = f"{prefix} the answer to the question given the context is {label}."
    return {"source": source, "target": target}

files = {}
for split in ["train", "dev", "test"]:
    for mode in ["source", "target", "id"]:
        file_name = f"pubmedqa-{split}-{mode}.txt"
        files[file_name] = open(file_name, "w")

for example_id in examples:
    biogpt_format = biogpt_version(examples[example_id])
    files[f"pubmedqa-{id_to_split[example_id]}-source.txt"].write(biogpt_format["source"]+"\n")
    files[f"pubmedqa-{id_to_split[example_id]}-target.txt"].write(biogpt_format["target"]+"\n")
    files[f"pubmedqa-{id_to_split[example_id]}-id.txt"].write(example_id+"\n")

for file_name in files:
    files[file_name].close()

