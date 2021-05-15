Model Overview
==============
Given an input sentence, list of mentions to be disambiguated, and list of possible candidates for each mention (described in `Input Data`_), Bootleg outputs the most likely candidate for each mention. Bootleg's model consists of three components: the embeddings, the embedding payload, and the attention network. For each entity candidate, the embedding component generates a set of relevant embeddings for each entity based on its types, relations, and other relevant metadata. The embedding payload merges these embedding components together into a single entity representation. This representation then gets used, along with our sentence embedding, as inputs into our neural model.

We now describe each step in detail and explain how to add/remove them from the our `Bootleg Config`_.

Bootleg Embeddings
------------------
All embeddings share a few common requirements. For one, any embedding must subclass our base embedding class at `bootleg/embeddings/base_emb.py <../apidocs/bootleg.embeddings.html#module-bootleg.embeddings.base_emb>`_ and define a ``forward`` function that takes as input candidate entity ids (size ``B x M x K`` with batch size ``B``, number of mentions ``M``, and number of candidates ``K``) and outputs an embedding, one for each candidate. There are a few caveats to this described in `Embedding Nuances`_. Secondly, all embeddings share a common set of configurable parameter that can be set in the config. These are

* ``load_class``: The class name of the embedding to load
* ``key``: A unique key for each embedding
* ``cpu``: True/False if embeddings should be placed on CPU (only valid for `bootleg/embeddings/entity_embs.py <../apidocs/bootleg.embeddings.html#module-bootleg.embeddings.entity_embs>`_)
* ``freeze``: True/False if all parameters of that embedding class should be frozen
* ``dropout1d``: 1D dropout percent to be applied before retuning the embedding from the forward
* ``dropout2d``: 2D dropout percent to be applied before retuning the embedding from the forward
* ``normalize``: True/False if embedding should be L2 normalized before retuning the embedding from the forward
* ``send_through_bert``: True/False if this embedding outputs token ids to be sent through BERT (see `Embedding Nuances`_)
* ``args``: JSON of custom arguments for the embedding passed in the ``init``.

An example YAML config setting these in the ``data_confg.ent_embeddings`` field looks like

.. code-block::

    data_config:
        ...
        ent_embeddings:
            - load_class: LearnedEntityEmb
              key: learned
              cpu: false
              freeze: false
              dropout1d: 0.0
              dropout2d: 0.4
              normalize: true
              sent_through_bert: false
              args:
                  learned_embedding_size: 256
            - load_class: LearnedTypeEmb
              key: learned_type
              ...


Entity Embeddings
^^^^^^^^^^^^^^^^^
Entity embeddings (see `bootleg/embeddings/entity_embs.py <../apidocs/bootleg.embeddings.html#module-bootleg.embeddings.entity_embs>`_) are an embedding that is unique for each entity. We support three kinds of entity embeddings: a learned entity embedding (class ``LearnedEntityEmb``), a subselected learned entity embeddings where only the most popular entity embeddings get an embedding (class ``TopKEntityEmb``), and a static embedding that loads an array of preexisting embeddings for each entity (class ``StaticEmb``).

Required config parameters passed in ``args`` are

``LearnedEntityEmb``

* ``learned_embedding_size``: dimension of embedding

``TopKEntityEmb``

* ``learned_embedding_size``: dimension of embedding
* ``perc_emb_drop``: percent of embeddings to be dropped
* ``qid2topk_eid``: JSON file of QID -> topK entity ID (see `bootleg/utils/postprocessing/compress_topk_entity_embeddings.py <../apidocs/bootleg.utils.postprocessing.html#module-bootleg.utils.postprocessing.compress_topk_entity_embeddings>`_)

``StaticEmb``

* ``emb_file``: path to embedding matrix or JSON with QID -> embedding array (see `bootleg/utils/preprocessing/build_static_embeddings.py <../apidocs/bootleg.utils.preprocessing.html#module-bootleg.utils.preprocessing.build_static_embeddings>`_ for an example embedding of the average title of an entity)

Type Embeddings
^^^^^^^^^^^^^^^^^
Type embeddings (see `bootleg/embeddings/type_embs.py <../apidocs/bootleg.embeddings.html#module-bootleg.embeddings.type_embs>`_) are unqiue for each type (class ``LearnedTypeEmb``). We get an embedding for each entity type and merge the type together via either averaging (default) or an additive attention (set via ``merge_func`` param being ``average`` or ``addattn``).

Required config parameters passed in ``args`` are

``LearnedTypeEmb``

* ``type_labels``: file (located inside ``data_config.emb_dir``) of QID -> list of type IDs
* ``type_dim``: type embedding dimension
* ``max_types``: maximum number of types to consider for each QID


KG Embeddings
^^^^^^^^^^^^^^^^^
KG embeddings (see `bootleg/embeddings/kg_embs.py <../apidocs/bootleg.embeddings.html#module-bootleg.embeddings.kg_embs>`_) are pairwise features that are used in our neural model. At their core, a KG embedding creates an adjacency matrix to be used in the model. We have two KG classes. The base KG embedding class returns the sum of the (possibly weighted) adjacency matrix of the batch candidates (class ``KGWeightedAdjEmb``). The final, does not return an actual embedding (described more in `Embedding Nuances`_) but instead directly creates a softmax weighted adjacency multiplication matrix to be used in the neural model (class ``KGIndices``).

Required config parameters passed in ``args`` are

``KGWeightedAdjEmb`` and ``KGIndices``

* ``kg_adj``: file (located inside ``data_config.emb_dir``) of either QID pairs, one per line (for binary adjacency) or JSON file with format head -> tail -> weight.

Title Embeddings
^^^^^^^^^^^^^^^^^
Title embeddings are embeddings based on the title of an entity. These are integrated into our BERT encoder so they are contextualized with each forward pass (especially useful if fine tuning BERT). See `Embedding Nuances`_ for details.

There are no required arguments for the title embedding. You will need to set ``send_through_bert: true``. Further, we recommend setting ``requires_grad: false`` as a custom ``arg``. This turns off gradient updates for the title embedding which can often be too expensive to store.

Type Prediction
------------------
We can optionally add a type prediction module to Bootleg. This uses the mention word embeddings to perform a soft type prediction and adds the predicted type embedding to each candidate for the predicted mention to the payload, described next. The type prediction is a secondary task learned during training (see `bootleg/layers/mention_type_prediction.py <../apidocs/bootleg.layers.html#module-bootleg.layers.mention_type_prediction>`_).

Embedding Payload
------------------
Our embedding payload is a simple concat and project of the embeddings into a hidden dimension. We do add in sine positional embeddings to the embeddings to indicate where the mention is in the sentence (see `bootleg/layers/embedding_payload.py <../apidocs/bootleg.layers.html#module-bootleg.layers.embedding_payload>`_)

Attention Network
------------------
Our attention network (see `bootleg/layers/attn_networks.py <../apidocs/bootleg.layers.html#module-bootleg.layers.attn_networks>`_) takes the sentence and embedding payload and outputs a score for each candidate for each mention.

We use `transformer <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_ modules to learn patterns over phrases and entities. Specifically, we have three core modules (and one optional):

#. Phrase Module: attention over the sentence and entity payloads
#. Co-Occurrence Module: self-attention over the entity payloads
#. KG Module: takes the sum of the output of the phrase and co-occurrence modules and leverages KG connectivity among candidates as weights in an attention
#. Mention Module (Optional): attention over the mention word embeddings and the candidates for that mention.

We use MLP softmax layers to score each mention and candidate independently, selecting the most likely candidate per mention.

.. _Input Data: input_data.html
.. _Bootleg Config: config.html
.. _Embedding Nuances: ../advanced/embedding_nuances.html
