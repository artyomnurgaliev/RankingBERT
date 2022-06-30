# Copyright 2022 Artyom Nurgaliev.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_ranking as tfr
from absl import flags
from official.nlp import optimization
from tensorflow_ranking.python.keras import network as tfrkeras_network

from config import bert_path, max_seq_length, batch_size, embedding_size

##########################################
# Model setup
##########################################
model = "emb"
input_package = f'tfrecords_{model}'
lr = 0.001
checkpoint_secs = 20
num_checkpoints = 100
num_train_steps = 40000
num_eval_steps = 200
loss = "approx_ndcg_loss"
list_size = 40
# output_model_dir = f"./naive_model/"
output_model_dir = f"./{model}_model/"
###########################################


flags.DEFINE_string("config", "cuda", "If 'cuda', running on GPU")

flags.DEFINE_bool("local_training", True, "If true, run training locally.")

flags.DEFINE_list("train_input_pattern", tf.io.gfile.glob(f'./{input_package}/train???.tfrecords'),
                  "Input file path pattern used for train.")

flags.DEFINE_list("eval_input_pattern", tf.io.gfile.glob(f'./{input_package}/test???.tfrecords'),
                  "Input file path pattern used for eval.")


flags.DEFINE_float("learning_rate", lr, "Learning rate for the optimizer.")

flags.DEFINE_integer("train_batch_size", batch_size,
                     "Number of input records used per batch for training.")

flags.DEFINE_integer("eval_batch_size", batch_size,
                     "Number of input records used per batch for eval.")

flags.DEFINE_integer("checkpoint_secs", checkpoint_secs,
                     "Saves a model checkpoint every checkpoint_secs seconds.")

flags.DEFINE_integer("num_checkpoints", num_checkpoints,
                     "Saves at most num_checkpoints checkpoints in workspace.")

flags.DEFINE_integer(
    "num_train_steps", num_train_steps,
    "Number of training iterations. Default means continuous training.")

flags.DEFINE_integer("num_eval_steps", num_eval_steps, "Number of evaluation iterations.")

flags.DEFINE_string(
    "loss", loss,
    "The RankingLossKey deciding the loss function used in training.")

flags.DEFINE_integer("list_size", list_size, "List size used for training.")

flags.DEFINE_bool("convert_labels_to_binary", False,
                  "If true, relevance labels are set to either 0 or 1.")

flags.DEFINE_string("model_dir", os.path.normpath(output_model_dir), "Output directory for models.")

# The followings are BERT related flags.
flags.DEFINE_string(
    "bert_config_file", bert_path + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. This "
    "specifies the model architecture. Please download the model from "
    "the link: https://github.com/google-research/bert")

flags.DEFINE_string(
    "bert_init_ckpt", bert_path + "bert_model.ckpt",
    "Initial checkpoint from a pre-trained BERT model. Please download from "
    "the link: https://github.com/google-research/bert")

flags.DEFINE_integer(
    "bert_max_seq_length", max_seq_length,
    "The maximum input sequence length (#words) after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

FLAGS = flags.FLAGS

_SIZE = "example_list_size"
_NETWORK_NAME = "emb"


def context_feature_columns():
    """Returns context feature names to column definitions."""
    return {}


def example_feature_columns():
    """Returns example feature names to column definitions.
  """
    feature_columns = {}
    feature_columns.update({
        "item_price":
            tf.feature_column.numeric_column(
                "item_price",
                shape=(1,),
                default_value=0,
                dtype=tf.float32),
        'image_shape':
            tf.feature_column.numeric_column(
                "image_shape",
                shape=(1,),
                default_value=0,
                dtype=tf.float32),
        'image_count':
            tf.feature_column.numeric_column(
                "image_count",
                shape=(1,),
                default_value=0,
                dtype=tf.float32),
        'title_emb':
            tf.feature_column.numeric_column(
                "title_emb",
                shape=(embedding_size,),
                default_value=0,
                dtype=tf.float32),
        'description_emb':
            tf.feature_column.numeric_column(
                "description_emb",
                shape=(embedding_size,),
                default_value=0,
                dtype=tf.float32),
        'brand_emb':
            tf.feature_column.numeric_column(
                "brand_emb",
                shape=(embedding_size,),
                default_value=0,
                dtype=tf.float32),
        'category_emb':
            tf.feature_column.numeric_column(
                "category_emb",
                shape=(embedding_size,),
                default_value=0,
                dtype=tf.float32),
    })
    return feature_columns


class EmbeddingRankingNetwork(tfrkeras_network.UnivariateRankingNetwork):
    def __init__(self,
                 context_feature_columns,
                 example_feature_columns,
                 name=_NETWORK_NAME,
                 **kwargs):
        """
        Args:
          context_feature_columns: A dict containing all the context feature columns
            used by the network. Keys are feature names, and values are instances of
            classes derived from `_FeatureColumn`.
          example_feature_columns: A dict containing all the example feature columns
            used by the network. Keys are feature names, and values are instances of
            classes derived from `_FeatureColumn`.
          name: name of Keras network.
          **kwargs: keyword arguments.
        """
        super(EmbeddingRankingNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            name=name,
            **kwargs)

        self._dropout_layer = tf.keras.layers.Dropout(
            rate=0.1)

        self._category_emb_dense1 = tf.keras.layers.Dense(50, activation='relu')
        self._category_emb_dense2 = tf.keras.layers.Dense(10, activation='relu')
        self._category_emb_dense3 = tf.keras.layers.Dense(1, activation='relu')

        self._features_emb_dense1 = tf.keras.layers.Dense(50, activation='relu')
        self._features_emb_dense2 = tf.keras.layers.Dense(10, activation='relu')
        self._features_emb_dense3 = tf.keras.layers.Dense(1, activation='relu')

        self._dense1 = tf.keras.layers.Dense(3, activation='relu')
        self._score_layer = tf.keras.layers.Dense(units=1, name="score")

    def score(self, context_features=None, example_features=None, training=True):
        """Univariate scoring of context and one example to generate a score.

        Args:
          context_features: (dict) Context feature names to 2D tensors of shape
            [batch_size, ...].
          example_features: (dict) Example feature names to 2D tensors of shape
            [batch_size, ...].
          training: (bool) Whether in training or inference mode.

        Returns:
          (tf.Tensor) A score tensor of shape [batch_size, 1].
        """

        def get_inputs():
            item_price = example_features["item_price"]
            image_shape = example_features["image_shape"]
            image_count = example_features["image_count"]

            title_emb = example_features["title_emb"]
            description_emb = example_features["description_emb"]
            brand_emb = example_features["brand_emb"]
            category_emb = example_features["category_emb"]

            features = tf.concat([item_price, image_shape, image_count], axis=1)

            features_emb = tf.concat([brand_emb, title_emb, description_emb], axis=1)

            category_emb = self._category_emb_dense1(category_emb)
            category_emb = self._category_emb_dense2(category_emb)

            features_emb = self._features_emb_dense1(features_emb)
            features_emb = self._features_emb_dense2(features_emb)

            mult = self._category_emb_dense3(category_emb) * self._features_emb_dense3(features_emb)
            result = tf.concat([mult, features_emb, category_emb, features], axis=1)
            # naive model:
            # result = tf.concat([features], axis=1)
            return result

        result = get_inputs()
        result = self._dropout_layer(result, training=training)
        result = self._dense1(result)
        result = self._dropout_layer(result, training=training)
        return self._score_layer(result)

    def get_config(self):
        config = super(EmbeddingRankingNetwork, self).get_config()
        return config


def get_estimator(hparams):
    """Create Keras ranking estimator."""
    network = EmbeddingRankingNetwork(
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        name=_NETWORK_NAME)

    loss = tfr.keras.losses.get(
        hparams.get("loss"),
        reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)

    metrics = tfr.keras.metrics.default_keras_metrics()

    config = tf.estimator.RunConfig(
        model_dir=hparams.get("model_dir"),
        keep_checkpoint_max=hparams.get("num_checkpoints"),
        save_checkpoints_secs=hparams.get("checkpoint_secs"))

    optimizer = optimization.create_optimizer(
        init_lr=hparams.get("learning_rate"),
        num_train_steps=hparams.get("num_train_steps"),
        num_warmup_steps=hparams.get("bert_num_warmup_steps"),
        optimizer_type="adamw")

    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        size_feature_name=_SIZE)

    return tfr.keras.estimator.model_to_estimator(
        model=ranker,
        model_dir=hparams.get("model_dir"),
        config=config
    )


def train_and_eval():
    hparams = dict(
        train_input_pattern=FLAGS.train_input_pattern,
        eval_input_pattern=FLAGS.eval_input_pattern,
        learning_rate=FLAGS.learning_rate,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        checkpoint_secs=FLAGS.checkpoint_secs,
        num_checkpoints=FLAGS.num_checkpoints,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        loss=FLAGS.loss,
        list_size=FLAGS.list_size,
        listwise_inference=True,  # Only supports `True` in keras Ranking Network.
        convert_labels_to_binary=FLAGS.convert_labels_to_binary,
        model_dir=FLAGS.model_dir,
        config=FLAGS.config,
    )

    ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        hparams=hparams,
        estimator=get_estimator(hparams),
        label_feature_name="rank",
        label_feature_type=tf.float32,
        size_feature_name=_SIZE)

    ranking_pipeline.train_and_eval(local_training=FLAGS.local_training)


def main(_):
    train_and_eval()


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_built_with_cuda())
    tf.compat.v1.app.run()
