# Biomedical downstream evaluation

## NLU
### Dependencies
```bash
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.9.1 datasets==1.11.0 wandb sklearn seqeval
```

### Usage
For PubMedQA, go to `seqcls/` and run the following command:
```bash
task=pubmedqa_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python3 -u run_seqcls_gpt.py --tokenizer_name gpt2 --model_name_or_path gpt2 \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 30  --max_seq_length 512  --logging_steps 100 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
```


For MedQA-USMLE, go to `mc/` and run the following command:
```bash
task=medqa_usmle_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python3 -u run_multiple_choice.py --tokenizer_name gpt2 --model_name_or_path gpt2 \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 30 --max_seq_length 512 --fp16 --logging_first_step --logging_steps 20 \
  --save_strategy no --evaluation_strategy steps --eval_steps 100 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
```


For NER tasks (EBM-PICO, BC5CDR-disease), go to `tokcls/` and run the following command:
```bash
task=ebmnlp_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python3 -u run_ner.py --tokenizer_name gpt2 --model_name_or_path gpt2 \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --warmup_steps 100 --num_train_epochs 1  --max_seq_length 512  --logging_steps 100 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  --return_macro_metrics \
  |& tee $outdir/log.txt &
```
```bash
task=BC5CDR-disease_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python3 -u run_ner.py --tokenizer_name gpt2 --model_name_or_path gpt2 \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 5e-5 --warmup_steps 100 --num_train_epochs 20  --max_seq_length 512  --logging_steps 100 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
```


## NLG
Go to `./textgen`.

### Files
    textgen
    ├── gpt2                          # Code for GPT2 style autoregressive LM
    │   ├── train_e2e.py              # high-level scripts to train.
    │   ├── train_control.py          # code that implements prefix-tuning.
    │   ├── trainer_prefix.py         # trainer code for the training loop.
    │   ├── run_language_modeling.py  # training code (contains data loading, model loading, and calls trainer)
    │   ├── gen.py                    # high-level scripts to decode.
    │   ├── run_generation.py         # decoding code.
    │   ├── gen_batch.py              # batched version.
    │   └── run_generation_batch.py   # batched version.

### Dependencies
```bash
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install sacrebleu==2.0.0 rouge-score==0.0.4
cd transformers; pip install -e .
```

### Usage (seq2seq tasks)
Make sure the task dataset is in `./textgen/data`. See `medparasimp` (a medical text simplification task) as an example. The dataset folder should have `<split>.source` and `<split>.target` files.

Go to `./textgen/gpt2`.
To finetune, run:
```
python -u train_e2e.py --mode medparasimp --tuning_mode finetune --epoch 20 --learning_rate 5e-5 --bsz 2 --gradient_accumulation_step 16 --seed 101 --model_name_or_path gpt2 --model_nickname gpt2 --warmup_steps 1000 &
```
You can also specify `--model_name_or_path` to be your custom model.

After finetuning, run generation on the test set by:
```
python gen_batch.py --mode medparasimp --batch_size 9 --length 400 --no_repeat_ngram_size 6 --control_mode no --use_prefixtuning 0 --eval_split test \
    --base_model_name_or_path gpt2 \
    --load_checkpoint_path <finetuned_model_path>
```
For example, `finetuned_model_path` can be `runs_medparasimp/medparasimpfinetune_gpt2_y_0_act_cat_b=32-e=20_d=0.0_u=no_lr=5e-05_wm=1000_w=0.0_s=101_r=n_m=512_uctd=no_o=1_o=1`.
This generation run will print out ROUGE evaluation score at the end.


### Acknowledgement
The NLG part of the code was built on https://github.com/XiangLi1999/PrefixTuning
