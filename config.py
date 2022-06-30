# Please download `bert_config_file` and `bert_init_ckpt` from tensorflow models
# website: https://github.com/tensorflow/models/tree/master/official/nlp/bert.
# Note that those checkpoints are TF 2.x compatible, which are different from the
# checkpoints downloaded here: https://github.com/google-research/bert. You may
# convert a TF 1.x checkpoint to TF 2.x using `tf2_encoder_checkpoint_converter`
# under https://github.com/tensorflow/models/tree/master/official/nlp/bert.
# The following command downloads an uncased BERT-base model checkpoint for you:
# mkdir /tmp/bert && \
# wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/\
# uncased_L-12_H-768_A-12.tar.gz -P /tmp/bert  && \
# tar -xvf /tmp/bert/uncased_L-12_H-768_A-12.tar.gz -C /tmp/bert/  && \
bert_path = "./uncased_L-2_H-128_A-2_TF2/"
IMG_HEIGHT = 40
IMG_WIDTH = 40
batch_size = 1
embedding_size = 100
max_seq_length = 128
