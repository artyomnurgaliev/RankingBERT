BERT for ranking

The problem of predicting the relevance of new products on the e-commerce platform is being solved. By the list of products and the request, a sorted list of products is displayed

The repository contains 4 models, solving ranking task:
* emb: pre-trained text embeddings
* conv: convolutional neural network for working with product images
* bert: BERT modified for ranking
* union: BERT + ConvNN

The work uses
[TF-Ranking](https://dl.acm.org/doi/abs/10.1145/3292500.3330677) for training models

How to train model:
1. To download data, use code in notebook: [downloading](data/loading%20dataset.ipynb)
2. For preprocessing, run the code in notebooks: [first phase](data/data%20preprocessing,%20first%20phase.ipynb), [second phase](data/data%20preprocessing,%20second%20phase.ipynb)
3. After preprocessing, the data must be converted to the tf-records format.
Each model has its own set of input data: To create tf-records, run code in [tfrecords.py](tfrecords.py)
You should select the model, for which records are generated, `model` can be `emb`, `conv`, `bert` and `union`
If flag `write_records` is set to `True`, than records will be written to the path `output_dir` 
4. To train model, one can use `train_<model>.py` file, for example, to train BERT, use [code](train_bert.py).
   (for training BERT and union model you will need checkpoints, look at info in the [config](config.py) file)

Graduate work  
MIPT  
Student: Artyom Nurgaliev  
Academic advisor: Petr Bolotin  
Department of Computational Linguistics  

