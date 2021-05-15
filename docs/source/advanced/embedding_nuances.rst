Embedding Nuances
==================
As mention in `Bootleg Model`_, there are two edge cases of how we use embeddings. The first is our title embedding being integrated in the BERT encoder and the second is how we allow for our ``KGIndices`` KG embedding to pass the adjacency matrix to the attention network to allow for a KG-relation model component.

Title Embeddings and BERT
--------------------------
We now describe our title embedding, but the notes here apply to _any_ embedding that sets ``send_through_bert: true`` in the embedding config.

As our title embedding requires being sent through BERT to get the embeddings associated with each entity title, it will function a little different than our other embeddings. In particular, the ``forward`` pass now returns token ids to be passed to BERT. It can optionally return the token type ids, attention mask, a boolean if gradients should be accumulated or not, and another other forward values. Up to the first four outputs (token ids, token type ids, attention mask, and gradient boolean) will be send through BERT. Any remaining inputs will be passed to the postprocessing method, described next.

After getting the BERT word embeddings, these may need to be postprocessed (e.g., normalized, projected, ...). As such, each title embedding must define a ``postprocess_embedding`` function. This will take as input the title embedding output, the output token mask, and any remaining outputs from the ``forward`` beyond the first four. This postprocess function will return the final embedding matrix for each entity candidate.


KG Batch on the Fly
--------------------
In order to allow for the KG adjacency matrix to be send through to the attention network, two things need to happen. The first is that, for each sentence of data, we need to extract the adjacency values for all entity candidates. While this would normally be done in a ``forward`` pass, as we store the KG adjacency as a sparse numpy matrix, we decide to push this computation in the ``__get_item__`` method of our dataset to take advantage of the dataloader's multiprocessing. This is what our ``batch_on_the_fly_data`` input dictionary is allowing for. We store the prepped KG information in this dictionary instead of creating it in our ``forward``.

Now, when we do call ``forward`` for our KG embeddings, it can use the unique embedding key to extract the KG data from ``batch_on_the_fly_data``. Some KG embedding classes will sum this value for each candidate. The ``KGIndices`` class will process this matrix to turn it into a weighted softmax matrix for multiplying the entity canddiates with in the neural model. To send this matrix to the neural model, we again use the ``batch_on_the_fly_data`` dictionary and story the processed matrix in that dictionary to be accessed in the neural model.


.. _Bootleg Model: ../gettingstarted/model.html
