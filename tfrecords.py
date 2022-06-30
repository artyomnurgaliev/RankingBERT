import os
import gensim

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from official.nlp.bert import tokenization
from tensorflow_serving.apis import input_pb2
import string
from nltk.corpus import stopwords
from string import punctuation
import nltk
from config import max_seq_length, bert_path, IMG_HEIGHT, IMG_WIDTH
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

EPS = 1e-5

class EncoderWord2Vec:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.wv.vectors[0])

    def encode_sent(self, words) -> np.ndarray:
        return np.mean([self.word2vec.wv.get_vector(w) for w in words if w in self.word2vec.wv]
                or [np.zeros(self.dim)], axis=0)


corpus = api.load('text8')
model = Word2Vec(corpus)
WORD2VEC = EncoderWord2Vec(model)

nltk.download('stopwords')
NOISE = set(stopwords.words('english') + list(punctuation))

NORMALIZATION_LAYER = tf.keras.layers.Rescaling(1./255)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    try:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    except:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    try:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    except:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    except:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def category_example(category_name):
    feature = {
        'category_name': _bytes_feature(bytes(str(category_name), 'utf-8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def truncate_seq(tokens_sequences, max_length):
    """Truncates a list of sequences with tokens in place to the maximum length."""
    assert max_length > 0
    assert isinstance(max_length, int)
    assert len(tokens_sequences) > 0
    total_length = np.sum([len(seq) for seq in tokens_sequences])
    curr_length = 0
    if total_length > max_length:
        # Truncation is needed.
        for i, seq in enumerate(tokens_sequences):
            del seq[(max_length - curr_length) // (len(tokens_sequences) - i):]
            curr_length += len(seq)


def to_bert_ids(tokenizer, sequences, max_length):
    """Converts a list of sentences to related Bert ids.

    Args:
      tokenizer - Bert tokenizer
      sequences - Iterable of sequences

    Returns:
      A tuple (`input_ids`, `input_masks`, `segment_ids`) for Bert finetuning.
    """
    assert len(sequences) > 0
    tokens_sequences = [tokenizer.tokenize(seq) for seq in sequences]
    assert max_length > 0
    truncate_seq(tokens_sequences, max_length - (len(sequences) + 1))

    # The convention in BERT for sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP] got it [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1     0   0   0
    #
    # The `type_ids` (aka. `segment_ids`) are used to indicate whether this is
    # the first sequence, the second sequence and etc. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    #
    # When there is only one sentence given, the sequence pair would be:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0
    #

    tokens = ["[CLS]"]
    for seq in tokens_sequences:
        tokens = tokens + seq + ["[SEP]"]

    segment_ids = [0] + [0] * len(tokens_sequences[0]) + [0]
    for i, seq in enumerate(tokens_sequences):
        if i != 0:
            segment_ids = segment_ids + [i % 2] * len(seq) + [i % 2]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_length:
        padding_len = max_length - len(input_ids)
        input_ids.extend([0] * padding_len)
        input_mask.extend([0] * padding_len)
        segment_ids.extend([0] * padding_len)

    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    assert len(segment_ids) == max_length

    return input_ids, input_mask, segment_ids


def item_example_emb(row, category_name):
    title = row.title
    description = row.description
    brand = row.brand
    price = row.price
    rank = row.rank_scaled
    url_high_res = row.image_url_high_res
    image_count = row.image_count

    image_shape = 0

    if image_count != 0:
        image_path_high_res = "./data/" + url_high_res
        try:
            image = tf.io.decode_jpeg(tf.io.read_file(image_path_high_res))
            image_shape = image.shape[0]
        except:
            print("skipped")
            pass

    res = {
        'item_price': price,
        'rank': rank,
        'image_shape': image_shape,
        'image_count': image_count,
        'title_emb': create_emb(title),
        'description_emb': create_emb(description),
        'brand_emb': create_emb(brand),
        'category_emb': create_emb(category_name),
    }
    return res


def item_example_conv(row, category_name):
    price = row.price
    rank = row.rank_scaled
    url_high_res = row.image_url_high_res
    image_count = row.image_count

    image_shape = 0
    image = np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float)

    if image_count != 0:
        image_path_high_res = "./data/" + url_high_res
        try:
            image = tf.io.decode_jpeg(tf.io.read_file(image_path_high_res))
            image_shape = image.shape[0]
            image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
            image = NORMALIZATION_LAYER(image)
            image = np.array(image).reshape(IMG_HEIGHT * IMG_WIDTH * 3)
        except:
            print("skipped")
            image = np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float)

    res = {
        'item_price': price,
        'rank': rank,
        'image_shape': image_shape,
        'image_count': image_count,
        'image': image,
        'category_emb': create_emb(category_name)
    }
    return res


def item_example_bert(row, tokenizer, category_name):
    title = row.title
    description = row.description
    brand = row.brand
    price = row.price
    rank = row.rank_scaled
    url_high_res = row.image_url_high_res
    image_count = row.image_count

    description = description.translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()
    title = title.translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()

    (input_ids, input_mask, segment_ids) = to_bert_ids(tokenizer,
                                                       [category_name, brand, title, description],
                                                       max_length=max_seq_length)

    image_shape = 0

    if image_count != 0:
        image_path_high_res = "./data/" + url_high_res
        try:
            image = tf.io.decode_jpeg(tf.io.read_file(image_path_high_res))
            image_shape = image.shape[0]
        except:
            print("skipped")

    res = {
        'item_price': price,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'rank': rank,
        'image_shape': image_shape,
        'image_count': image_count,
        'category_emb': create_emb(category_name)
    }
    return res


def item_example_union(row, tokenizer, category_name):
    title = row.title
    description = row.description
    brand = row.brand
    price = row.price
    rank = row.rank_scaled
    url_high_res = row.image_url_high_res
    image_count = row.image_count

    description = description.translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()
    title = title.translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()

    (input_ids, input_mask, segment_ids) = to_bert_ids(tokenizer,
                                                       [category_name, brand, title, description],
                                                       max_length=max_seq_length)

    image_shape = 0
    image = np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float)

    if image_count != 0:
        image_path_high_res = "./data/" + url_high_res
        try:
            image = tf.io.decode_jpeg(tf.io.read_file(image_path_high_res))
            image_shape = image.shape[0]
            image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
            image = NORMALIZATION_LAYER(image)
            image = np.array(image).reshape(IMG_HEIGHT * IMG_WIDTH * 3)
        except:
            print("skipped")
            image = np.zeros([IMG_HEIGHT * IMG_WIDTH * 3]).astype(np.float)

    res = {
        'item_price': price,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'rank': rank,
        'image_shape': image_shape,
        'image_count': image_count,
        'image': image,
        'category_emb': create_emb(category_name)
    }
    return res

def create_emb(sentence):
    import razdel

    filtered = []
    for token in razdel.tokenize(sentence):
        word = token.text.lower()
        if word not in NOISE:
            filtered.append(word)

    return np.array(WORD2VEC.encode_sent(filtered))


def normalize(examples, column_name):
    new_examples = []
    values = []
    for example in examples:
        if column_name not in example.keys():
            return examples
        if np.isnan(example[column_name]):
            continue
        values.append(example[column_name])

    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    for example in examples:
        example[column_name] = (example[column_name] - mean) / (std + EPS)
        new_examples.append(example)

    return new_examples


def process_df(df, tokenizer, category_name, model=model):
    """
    Takes pandas dataframe and returns list of Examples
    """
    examples = []

    for i in range(df.shape[0]):
        row = df.iloc[i]
        if model == "emb":
            examples.append(item_example_emb(row, category_name))
        if model == "bert":
            examples.append(item_example_bert(row, tokenizer, category_name))
        if model == "conv":
            examples.append(item_example_conv(row, category_name))
        if model == "union":
            examples.append(item_example_union(row, tokenizer, category_name))

    features = []
    # normalization
    examples = normalize(examples, "item_price")
    examples = normalize(examples, "image_shape")
    examples = normalize(examples, "image_count")
    # examples = normalize(examples, "rank")

    for example in examples:
        feature = {}
        for k in example.keys():
            if k in ['input_ids', 'input_mask', 'segment_ids']:
                feature[k] = _int64_feature(example[k])
            else:
                feature[k] = _float_feature(example[k])
        feature = tf.train.Example(features=tf.train.Features(feature=feature))
        features.append(feature)
    return features


def create_records(df, tokenizer, output_dir, num_of_records=5, prefix=None, model="emb"):
    """
    Takes a pandas dataframe and number of records to create and creates TFRecords.
    Saves records in output_dir
    """
    df = df.fillna('')

    all_categories = list(set(df.category_id.values.tolist()))

    record_prefix = os.path.join(output_dir, prefix)
    files_per_record = int(len(all_categories) / num_of_records)  # approximate number of examples per record
    chunk_number = 0

    for i in range(0, len(all_categories), files_per_record):
        print("Writing chunk ", str(chunk_number))
        category_chunk = all_categories[i:i + files_per_record]

        if num_of_records == 1:
            record_file = record_prefix + ".tfrecords"
        else:
            record_file = record_prefix + str(chunk_number).zfill(3) + ".tfrecords"

        with tf.io.TFRecordWriter(record_file) as writer:
            for category in tqdm.tqdm(category_chunk):
                category_df = df.loc[df["category_id"] == category]
                category_name = category_df["category"].values.tolist()[0]

                # actually unused, because we add context to each item example
                context = category_example(category_name)

                examples = process_df(category_df, tokenizer, category_name, model=model)

                elwc = input_pb2.ExampleListWithContext()
                elwc.context.CopyFrom(context)
                for example in examples:
                    example_features = elwc.examples.add()
                    example_features.CopyFrom(example)

                writer.write(elwc.SerializeToString())
            chunk_number += 1


def read_and_print_tf_record(target_filename, num_of_examples_to_read):
    filenames = [target_filename]
    tf_record_dataset = tf.data.TFRecordDataset(filenames)
    all_examples = []

    for raw_record in tf_record_dataset.take(num_of_examples_to_read):
        example_list_with_context = input_pb2.ExampleListWithContext()
        example_list_with_context.ParseFromString(raw_record.numpy())
        all_examples.append(example_list_with_context)

    return all_examples


def main():
    train_data_path = "./data/train_with_images_small.csv"
    test_data_path = "./data/test_with_images_small.csv"
    # train_data_path = "./data/train_with_images.csv"
    # test_data_path = "./data/test_with_images.csv"

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    # output_dir = f"./tfrecords_with_images_{max_seq_length}/"

    model = "union" # "emb", "bert", "conv", "union"
    output_dir = f"./tfrecords_{model}/"

    write_records = True
    if write_records:
        tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_path, "vocab.txt"), do_lower_case=True)
        num_of_records = 10
        print("Creating records for train")
        create_records(train_df, tokenizer, output_dir, num_of_records, prefix="train", model=model)
        print("Creating records for test")
        create_records(test_df, tokenizer, output_dir, num_of_records, prefix="test", model=model)
    else:
        examples = read_and_print_tf_record(output_dir + "train000.tfrecords", 1)
        print(examples)


if __name__ == "__main__":
    main()
