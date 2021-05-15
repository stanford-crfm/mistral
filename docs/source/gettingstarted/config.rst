Configuring Bootleg
====================

By default, Bootleg loads the default config from `bootleg/utils/parser/bootleg_args.py <../apidocs/bootleg.utils.parser.html#module-bootleg.utils.parser.bootleg_args>`_. When running a Bootleg model, the user may pass in a custom JSON or YAML config via::

  python3 bootleg/run.py --config_script <path_to_config>

This will override all default values. Further, if a user wishes to overwrite a param from the command line, they can pass in the value, using the dotted path of the argument. For example, to overwrite the data directory (the param ``data_config.data_dir``, the user can enter::

  python3 bootleg/run.py --config_script <path_to_config> --data_config.data_dir <path_to_data>

Bootleg will save the run config (as well as a fully parsed verison with all defaults) in the log directory.

Emmental Config
________________

As Bootleg uses Emmental_, the training parameters (e.g., learning rate) are set and handled by Emmental. We provide all Emmental params, as well as our defaults, at `bootleg/utils/parser/emm_parse_args.py <../apidocs/bootleg.utils.parser.html#module-bootleg.utils.parser.emm_parse_args>`_. All Emmental params are under the ``emmental`` configuration group. For example, to change the learning rate and number of epochs in a config, add

.. code-block::

  emmental:
    lr: 1e-4
    n_epochs: 10
  run_config:
    ...

You can also change Emmental params by the command line with ``--emmental.<emmental_param> <value>``.

Example Training Config
________________________
An example training config is shown below

.. code-block::

    emmental:
      lr: 2e-5
      n_epochs: 3
      evaluation_freq: 0.2
      warmup_percentage: 0.1
      lr_scheduler: linear
      log_path: logs/wiki
      l2: 0.01
      grad_clip: 1.0
      distributed_backend: nccl
      fp16: true
    run_config:
      eval_batch_size: 32
      dataloader_threads: 4
      dataset_threads: 50
    train_config:
      batch_size: 32
    model_config:
      hidden_size: 512
      num_heads: 16
      num_model_stages: 2
      ff_inner_size: 1024
      attn_class: BootlegM2E
    data_config:
      data_dir: bootleg-data/data/wiki
      data_prep_dir: prep
      emb_dir: bootleg-data/embs
      ent_embeddings:
           - key: learned
             load_class: LearnedEntityEmb
             freeze: false
             cpu: false
             args:
               learned_embedding_size: 200
               regularize_mapping: /data/data/wiki_title_0122/qid2reg_pow.csv # GENERATED IN bootleg/utils/preprocessing/build_regularization_mapping.py
           - key: title_static
             load_class: StaticEmb
             freeze: false # Freeze the projection layer or not
             cpu: false # Freeze projection layer or not
             args:
               emb_file: /data/data/wiki_title_0122/static_wiki_0122_title.pt # GENERATED IN bootleg/utils/preprocessing/build_static_embeddings.py
               proj: 256
           - key: learned_type
             load_class: LearnedTypeEmb
             freeze: false
             args:
               type_labels: hyena_types_1229.json
               max_types: 3
               type_dim: 128
               merge_func: addattn
               attn_hidden_size: 128
           - key: learned_type_wiki
             load_class: LearnedTypeEmb
             freeze: false
             args:
               type_labels: wikidata_types_1229.json
               max_types: 3
               type_dim: 128
               merge_func: addattn
               attn_hidden_size: 128
           - key: learned_type_relations
             load_class: LearnedTypeEmb
             freeze: false
             args:
               type_labels: kg_relation_types_1229.json
               max_types: 50
               type_dim: 128
               merge_func: addattn
               attn_hidden_size: 128
           - key: adj_index
             load_class: KGIndices
             batch_on_the_fly: true
             normalize: false
             args:
               kg_adj: kg_adj_1229.txt
      entity_dir: bootleg-data/data/wiki/entity_db
      word_embedding:
        cache_dir: bootleg-data/embs/pretrained_bert_models
        freeze: true # FINE TUNE BERT OR NOT
        layers: 12
        bart_model: bert-base-cased

Default Config
_______________
The default Bootleg config is shown below

.. literalinclude:: ../../../bootleg/utils/parser/bootleg_args.py


.. _Emmental: https://github.com/SenWu/Emmental
