import json
import os
import shutil

from src.corpora import get_auto_dataset
from src.models import get_auto_clm_tokenizer
from src.util import create_paths
from tests import MISTRAL_TEST_DIR, run_tests


run_dir = f"{MISTRAL_TEST_DIR}/runs"
artifacts_dir = f"{MISTRAL_TEST_DIR}/data-preprocessing-test-artifacts"
model_config_path = "conf/models/gpt2-micro.json"

seq_len = 256

gold_example_input_ids = [
    796,
    5199,
    347,
    2852,
    353,
    796,
    198,
    50256,
    5199,
    347,
    2852,
    353,
    318,
    281,
    3594,
    2646,
    11,
    5581,
    290,
    21421,
    8674,
    13,
    679,
    550,
    257,
    8319,
    12,
    7364,
    1806,
    2597,
    319,
    262,
    5581,
    2168,
    383,
    3941,
    287,
    4751,
    13,
    770,
    373,
    3940,
    416,
    257,
    20495,
    2597,
    287,
    262,
    711,
    2332,
    684,
    3194,
    416,
    11288,
    37072,
    11,
    543,
    373,
    6157,
    287,
    5878,
    379,
    262,
    8111,
    3078,
    15752,
    13,
    679,
    550,
    257,
    8319,
    2597,
    287,
    262,
    5581,
    2168,
    8974,
    1757,
    1024,
    276,
    287,
    6244,
    13,
    554,
    5472,
    347,
    2852,
    353,
    11406,
    257,
    2597,
    355,
    366,
    40441,
    1,
    287,
    262,
    4471,
    366,
    51,
    21874,
    338,
    8362,
    1,
    286,
    262,
    5581,
    2168,
    383,
    5882,
    31623,
    26,
    339,
    31636,
    7848,
    10544,
    2940,
    13535,
    290,
    20893,
    12806,
    72,
    13,
    679,
    373,
    3350,
    287,
    262,
    5075,
    21421,
    32260,
    286,
    262,
    14576,
    39616,
    711,
    21673,
    22384,
    11,
    543,
    373,
    6157,
    379,
    262,
    25331,
    15752,
    287,
    42125,
    290,
    262,
    6065,
    959,
    24777,
    19239,
    287,
    3576,
    13,
    679,
    373,
    7924,
    416,
    1757,
    40928,
    290,
    31636,
    7848,
    3932,
    854,
    680,
    707,
    11,
    24379,
    1168,
    7056,
    11,
    5850,
    8758,
    11,
    28059,
    13709,
    411,
    11,
    35331,
    36442,
    290,
    36401,
    4789,
    13,
    198,
    50256,
    554,
    4793,
    11,
    347,
    2852,
    353,
    31636,
    7848,
    854,
    680,
    707,
    287,
    262,
    711,
    47002,
    3194,
    416,
    2940,
    12552,
    12639,
    13,
    679,
    4120,
    319,
    257,
    4793,
    4471,
    286,
    262,
    5581,
    2168,
    11,
    28274,
    11,
    3940,
    416,
    257,
    2597,
    287,
    262,
    4343,
    21421,
    3227,
    286,
    1374,
    284,
    19739,
    7924,
    416,
    22568,
    494,
    371,
    49003,
    13,
    1374,
    284,
    19739,
    373,
    6157,
    379,
    5511,
    15752,
    287,
    262,
    3576,
    48114,
]

gold_example_attention_mask = [1] * seq_len


def test_add_eos_token():
    # clear artifacts dir
    shutil.rmtree(artifacts_dir) if os.path.exists(artifacts_dir) else None

    # set test configurations for building tokenizer
    paths = create_paths("test-data-preprocessing", "gpt2-small", run_dir, artifacts_dir)

    with open(model_config_path) as f:
        model_configs = json.load(f)

    # get model and tokenizer
    model, tokenizer = get_auto_clm_tokenizer(
        "gpt2-small",
        paths,
        model_configs=model_configs,
        gradient_checkpointing=False,
        gc_checkpoint_every=-1,
        use_pretrained_tokenizer=True,
    )

    # create the dataset
    lm_dataset = get_auto_dataset(
        tokenizer,
        paths,
        dataset_id="wikitext",
        dataset_name="wikitext-2-raw-v1",
        seq_len=seq_len,
        preprocessing_num_proc=4,
    )

    # check standard example from test is correct
    assert lm_dataset["test"][0]["input_ids"] == gold_example_input_ids
    assert lm_dataset["test"][0]["attention_mask"] == gold_example_attention_mask

    # check input_ids and attention mask have same length for each doc
    for doc in lm_dataset["train"]:
        assert len(doc["input_ids"]) == len(doc["attention_mask"])
        assert len(doc["input_ids"]) == seq_len

    # these are stats for wikitext2
    # grouping code processes batches of 1000
    # first 1000 docs have: 59112 tokens
    # first 1000 docs have: 647 nonempty docs
    # this produces: 59759 // 256 = 233 sequences
    # and throws out: 111 tokens
    # doc_id 998 tokens = 135 and doc_id 999 tokens = 83
    # so 2 <|endoftext|> will be thrown out amongst the 111
    # so we should see 645 <|endoftext|> tokens in the 233 sequences formed from first 1000 docs
    first_doc_batch = lm_dataset["train"].select([i for i in range(233)])
    num_doc_tokens = sum([1 for doc in first_doc_batch for i in doc["input_ids"] if i == tokenizer.eos_token_id])
    assert num_doc_tokens == 645

    # test contents for first 1000 docs from wikitext2 train is as expected (233 sequences)
    first_1000_gold = [idx for idx in open("data/first_1000_docs_tokens.txt").read().strip().split("\n")]
    first_1000 = [str(idx) for doc in lm_dataset["train"].select([i for i in range(233)]) for idx in doc["input_ids"]]
    assert first_1000_gold == first_1000

    # test contents for second 1000 docs from wikitext2 train is as expected (276 sequences)
    second_1000_gold = [idx for idx in open("data/second_1000_docs_tokens.txt").read().strip().split("\n")]
    second_1000 = [
        str(idx) for doc in lm_dataset["train"].select([i for i in range(233, 509)]) for idx in doc["input_ids"]
    ]
    assert second_1000_gold == second_1000


if __name__ == "__main__":
    run_tests()
