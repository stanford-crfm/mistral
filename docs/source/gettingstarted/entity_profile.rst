Entity Profiles
=================
Bootleg uses Wikipedia and Wikidata to collect and generate a entity database of metadata associated with an entity. We support both non-structural data (e.g., the title of an entity) and structural data (e.g., the type or relationship of an entity). We now describe how to generate entity profile data from scratch to be used for training and the structure of the profile data we already provide.

Generating Profiles
--------------------
The database of entity data starts with a simple ``jsonl`` file of data associated with an entity. Specifically, each line is a JSON object

.. code-block:: JSON

    {
        "entity_id": "Q16240866",
        "mentions": [["benin national u20 football team",1],["benin national under20 football team",1]],
        "title": "Forbidden fruit",
        "types": {"hyena": ["<wordnet_football_team_108080025>"],
                  "wiki": ["national association football team"],
                  "relations":["country for sport","sport"]},
        "relations": [
            {"relation":"P1532","object":"Q962"},
        ],
    }

The ``entity_id`` gives a unique string identifier of the entity. It does *not* have to start with a ``Q``. As we normalize to Wikidata, our entities are referred to as QIDs. The ``mentions`` provides a list of known aliases to the entity and a prior score associated with that mention indicating the strength of association. The score is used to order the candidates. The ``types`` provides the different types and entity is and supports different type systems. In the example above, the two type systems are ``hyena`` and ``wiki``. We also have a ``relations`` type system which treats the relationships an entity participates in as types. The ``relations`` JSON field provides the actual KG relationship triples where ``entity_id`` is the head.

.. note::

    By default, Bootleg assigns the score for each mentions as being the global entity count in Wikipedia. We empirically found this was a better scoring method for incorporating Wikidata "also known as" aliases that did not appear in Wikipedia. This means the scores for the mentions for a single entity will be the same.

We provide a more complete `sample of raw profile data <https://github.com/HazyResearch/bootleg/tree/master/data/sample_raw_entity_data/raw_profile.jsonl>`_ to look at.

Once the data is ready, we provide an `EntityProfile <../apidocs/bootleg.symbols.html#module-bootleg.symbols.entity_profile>`_ API to build and interact with the profile data. To create an entity profile for the model from the raw ``jsonl`` data, run

.. code-block:: python

    from bootleg.symbols.entity_profile import EntityProfile
    path_to_file = "data/sample_raw_entity_data/raw_profile.jsonl"
    # edit_mode means you are allowed to modify the profile
    ep = EntityProfile.load_from_jsonl(path_to_file, edit_mode=True)

.. note::

    By default, we assume that each alias can have a maximum of 30 candidates, 10 types, and 100 connections. You can change these by adding ``max_candidates``, ``max_types``, and ``max_connections`` as keyword arguments to ``load_from_jsonl``. Note that increasing the number of maximum candidates increases the memory required for training and inference.

Profile API
--------------------
Now that the profile is loaded, you can interact with the metadata and change it. For example, to get the title and add a type mapping, you'd run

.. code-block:: python

    ep.get_title("Q16240866")
    # This is adding the type "country" to the "wiki" type system
    ep.add_type("Q16240866", "country", "wiki")

Once ready to train or run a model with the profile data, simply save it

.. code-block:: python

    ep.save("data/sample_saved_entity_db")

We have already provided the saved dump at ``data/sample_entity_data``.

See our `entity profile tutorial <https://github.com/HazyResearch/bootleg/tree/master/tutorials/entity_profile_tutorial
.ipynb>`_ for a more complete walkthrough notebook of the API.

Training with a Profile
------------------------
Inside the saved folder for the profile, all the mappings needed to run a Bootleg model are provided. There are three subfolders as described below. Note that we use the word ``alias`` and ``mention`` interchangeably.

* ``entity_mappings``: This folder contains non-structural entity data.
    * ``qid2eid.json``: This is a mapping from entity id (we refer to this as QID) to an entity index used internally to extract embeddings. Note that these entity ids start at 1 (0 index is reserved for a "not in candidate list" entity). We use Wikidata QIDs in our tutorials and documentation but any string identifier will work.
    * ``qid2title.json``: This is a mapping from entity QID to entity Wikipedia title.
    * ``alias2qids.json``: This is a mapping from possible mentions (or aliases) to a list possible candidates. We restrict our candidate lists to be a predefined max length, typically 30. Each item in the list is a pair of [QID, QID score] values. The QID score is used for sorting candidates before filtering to the top 30. The scores are otherwise not used in Bootleg. This mapping is mined from both Wikipedia and Wikidata (reach out with a github issue if you want to know more).
    * ``alias2id.json``: This is a mapping from alias to alias index used internally by the model.
    * ``config.json``: This gives metadata associated with the entity data. Specifically, the maximum number of candidates.
* ``type_mappings``: This folder contains type entity data for each type system subfolder. Inside each subfolder are the following files.
    * ``type_vocab.json``: Mapping from type name to internal type id. This id mapping is offset by 1 to reserve the 0 type id for the UNK type.
    * ``qid2typenames.json``: Mapping from entity QID to a list of type names.
    * ``qid2typeids.json``: Mapping from entity QID to a list of type ids.
    * ``config.json``: Contains metadata of the maximum number of types allowed for an entity.
* ``kg_mappings``: This folder contains relationship entity data.
    * ``type_vocab.json``: Mapping from type name to internal type id. This id mapping is offset by 1 to reserve the 0 type id for the UNK type.
    * ``qid2relations.json``: Mapping from head entity QID to a dictionary of relation -> list of tail entities.
    * ``kg_adj.txt``: List of all connected entities separated by a tab. This is an unlabeled adjacency matrix.
    * ``config.json``: Contains metadata of the maximum number of tail connections allowed for a particular head entity and relation.

.. note::

    In Bootleg, we treat the relationships an entity participates in, whether as a head or tail entity, as types and use the unlabeled adjacency matrix as the KG connections in the model. This means one of our type systems is ``relations``.

.. note::

    In our public ``entity_db`` provided to run Bootleg models, we also provide a few extra files. The first is ``alias2qids_unfiltered.json`` which provides our unfiltered, raw candidate mappings. We filter noisy aliases before running mention extraction. We lastly provide ``type_vocab_to_wikidataqid.json`` in the ``wiki`` type system folder which maps our type names to their own Wikidata QIDs (all Wikidata types *are* QIDs).

Given this metadata, you simply need to specify the type, relation mappings and correct folder structures in a Bootleg training `config <config.html>`_. Specifically, these are the config parameters that need to be set to be associated with an entity profile.

.. code-block::

    data_config:
      emb_dir: data/sample_entity_data
      entity_dir: data/sample_entity_data
      ent_embeddings:
           - key: learned_type
             load_class: LearnedTypeEmb
             args:
               type_labels: type_mappings/wiki/qid2typeids.json
               type_vocab: type_mappings/wiki/type_vocab.json
           ...
           - key: adj_index
             load_class: KGIndices
             args:
               kg_adj: kg_mappings/kg_adj.txt

See our `example config <https://github.com/HazyResearch/bootleg/tree/master/configs/tutorial/sample_config.yaml>`_
for a full reference, and see our `entity profile tutorial <https://github
.com/HazyResearch/bootleg/tree/master/tutorials/entity_profile_tutorial.ipynb>`_ for some methods to help modify
configs
to map to the entity profile correctly.
