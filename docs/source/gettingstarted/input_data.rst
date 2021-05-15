Inputs
==============
Given an input sentence, Bootleg outputs the entities that participate in the text. For example, given the sentence

``Where is Lincoln in Logan County``

Bootleg should output that Lincoln refers to Lincoln IL and Logan County to Logan County IL.

This disambiguation occurs in two parts. The first, described here, is mention extraction and candidate generation, where phrases in the input text are extracted to be disambiguation. For example, in the sentence above, the phrases "Lincoln" and "Logan County" should be extracted. Each phrase to be disambiguated is called a mention (or alias). Instead of disambiguating against all entities in Wikipedia, Bootleg uses predefined candidate maps that provide a small subset of possible entity candidates for each mention. The second step, described in `Bootleg Model`_, is the disambiguation using Bootleg's neural model.

To understand how we do mention extraction and candidate generation, we first need to describe the profile data we have associated with an entity. Then we will describe how we perform mention extraction. Finally, we will provide details on the input data provided to Bootleg. Take a look at our `tutorials <https://github.com/HazyResearch/bootleg/tree/master/tutorials>`_ to see it in action.

Entity Data
--------------------
Bootleg uses Wikipedia and Wikidata to collect and generate a entity database of metadata associated with an entity. This is all located in ``entity_db`` and contains mappings from entities to structural data and possible mention. We describe the entity profiles in more details and how to generate them on our `entity profile <entity_profile.html>`_ page. For reference, we have an `EntityProfile <../apidocs/bootleg.symbols.html#module-bootleg.symbols.entity_profile>`_ class that loads and manages this metadata.

As our profile data does give us mentions that are associated with each entity, we now need to describe how we generate mentions.

Mention Extraction
------------------
Our mention extraction is a simple n-gram search over the input sentence (see `bootleg/end2end/extract_mentions.py <../apidocs/bootleg.end2end.html#module-bootleg.end2end.extract_mentions>`_). Starting from the largest possible n-grams and working towards single word mentions, we iterate over the sentence and see if any n-gram is a hit in our ``alias2qid`` mapping. If it is, we extract that mention. This enusre that each mention has a set of candidates.

To prevent extracting noisy mentions, like the word "the", we filter our alias maps to only have words that appear approximately more that 1.5% of the time as mentions in our training data.

The input format is in ``jsonl`` format where each line is a json object of the form

* ``sentence``: input sentence.

We output a jsonl with

* ``sentence``: input sentence.
* ``aliases``: list of extracted mentions.
* ``spans``: list of word offsets [inclusive, exclusive) for each alias.

Textual Input
------------------
Once we have mentions and candidates, we are ready to run our Bootleg model. The raw input format is in ``jsonl`` format where each line is a json object. We have one json per sentence in our training data with the following files

* ``sentence``: input sentence.
* ``sent_idx_unq``: unique sentence index.
* ``aliases``: list of extracted mentions.
* ``qids``: list of gold entity id (if known). We use canonical Wikidata QIDs in our tutorials and documentation, but any id used in the entity metadata will work. The id can be ``Q-1`` if unknown, but you _must_ provide gold QIDs for training data.
* ``spans``: list of word offsets [inclusive, exclusive) for each alias.
* ``gold``: list of booleans if the alias is a gold anchor link from Wikipedia or a weakly labeled link.
* ``slices``: list of json slices for evaluation. See `advanced training <../advanced/distributed_training.html>`_ for details.

For example, the input for the sentence above is

.. code-block:: JSON

    {
        "sentence": "Where is Lincoln in Logan County",
        "sent_idx_unq": 0,
        "aliases": ["lincoln", "logan county"],
        "qids": ["Q121", "Q???"],
        "spans": [[2,3], [4,6]],
        "gold": [True, True],
        "slices": {}
    }

For more details on training, see our `training tutorial <training.rst>`_.

.. _Bootleg Model: model.html
.. _tutorials: tutorials.html
.. _Emmental: https://github.com/SenWu/Emmental
