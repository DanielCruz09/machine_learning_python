'''Loading and Preprocessing Data with TensorFlow'''

'''--------------------------------------------The tf.data API--------------------------------------'''

import tensorflow as tf

x = tf.range(10) # any data tensor
dataset = tf.data.Dataset.from_tensor_slices(x) # contains tensors 0...9

# for item in dataset:
#     print(item)

x_nested = {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]}
dataset = tf.data.Dataset.from_tensor_slices(x_nested)
# for item in dataset:
#     print(item)

#--------------------------------------Chaining Transformations----------------------------------
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
# for item in dataset:
#     print(item)

dataset = dataset.map(lambda x: x * 2) # x is a batch
# for item in dataset:
#     print(item)

dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
# dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
# for item in dataset:
#     print(item)

# for item in dataset.take(2): # take a look at the first 2 batches
#     print(item)

#------------------------------------Shuffling the Data-------------------------------------------

dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
# for item in dataset:
#     print(item)

#----------------------------Interleaving Lines from Multiple Lines-------------------------------

def interleave_lines(train_filepaths):
    filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
    n_readers = 5
    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers
    )
    for line in dataset.take(5):
        print(line)

'''-------------------------------------------Preprocessing the Data-------------------------------------------'''

import numpy as np
x_mean = np.random.rand() # mean of each feature, random in this case
x_std = np.random.rand() # scale of each feature, random in this case
n_inputs = 8

def parse_csv_line(line):
    # defs stores the decoded lines, missing values should default to zero
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    # tf.io_decode_csv() returns a list of scalar tensors
    fields = tf.io.decode_csv(line, record_defaults=defs)
    # We stack the scalar tensors into a 1D array
    return tf.stack(fields[:-1]), tf.stack(fields[-1:])

def preprocess(line):
    x, y = parse_csv_line(line)
    return (x - x_mean) / x_std, y

# print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))

def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None,
                       n_parse_threads=5, shuffle_buffer_size=10_100,
                       seed=42, batch_size=32):
    dataset = tf.dataset.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads
    )
    dataset = dataset.map(preprocess, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)

'''------------------------------------------Using the Dataset with Keras------------------------------------------------------'''

# Now, we can use the function to generate our preprocessed datasets
# train_set = csv_reader_dataset(train_filepaths)
# valid_set = csv_reader_dataset(valid_filepaths)
# test_set = csv_reader_dataset(test_filepaths)

# We build the model using the datasets
# model = tf.keras.Sequential([...])
# model.compile(loss="mse", optimizer="sgd")
# model.fit(train_set, validation_data=valid_set, epochs=5)

# test_mse = model.evaluate(test_set)
# new_set = test_set.take(3)  # pretend we have 3 new samples
# y_pred = model.predict(new_set)  # or you could just pass a NumPy array

@tf.function
def train_one_epoch(model, optimizer, loss_fn, train_set):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
# for epoch in range(n_epochs):
#     print("\rEpoch {}/{}".format(epoch + 1, n_epochs), end="")
#     train_one_epoch(model, optimizer, loss_fn, train_set)

'''-----------------------------------------The TFRecord Format--------------------------------------------'''

with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
# for item in dataset:
#     print(item)

#-----------------------------------------Compressed Files---------------------------------
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"Compress, compress, compress!")

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"], compression_type="GZIP")

#-------------------------------------------Protocol Buffers----------------------------------
from tensorflow._api.v2.train import BytesList, FloatList, Int64List
from tensorflow._api.v2.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "email": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }
    )
)

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    for _ in range(5):
        f.write(person_example.SerializeToString())

#-----------------------------------Loading and Parsing Examples------------------------------
        
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

def parse(serialized_example):
    return tf.io.parse_single_example(serialized_example, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).map(parse)
for parsed_example in dataset:
    tf.sparse.to_dense(parsed_example["emails"], default_value=b"") # convert regular tensor to sparse tensor
    # print(parsed_example["emails"].values)

def parse(serialized_example):
    return tf.io.parse_example(serialized_example, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(2).map(parse)
# for parsed_examples in dataset:
#     print(parsed_examples) # two examples at a time

#---------------------------------------Handling Lists of Lists-------------------------------

def parse_lists(serialized_sequence_example, context_feature_descriptions, sequence_feature_descriptions):
    parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
        serialized_sequence_example, context_feature_descriptions,
        sequence_feature_descriptions
    )
    parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
    return parsed_content

'''---------------------------------------------Keras Preprocessing Layers-------------------------------------------------------'''

#-----------------------------------The Normalization Layer---------------------------------------

def build_norm_model(train_data, valid_data):
    x_train, y_train = train_data
    x_valid, y_valid = valid_data

    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(x_train)
    x_train_scaled = norm_layer(x_valid)
    x_valid_scaled = norm_layer(x_valid)
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
    model.fit(x_train_scaled, y_train, epochs=5,
          validation_data=(x_valid_scaled, y_valid))
    
    final_model = tf.keras.Sequential([norm_layer, model])
    x_new = [...] # some new unscaled instances
    y_pred = final_model(x_new) # preprocess the data and makes predictions

    dataset = dataset.map(lambda x, y: (norm_layer(x), y))

class MyNormalization(tf.keras.layers.Layer):
    def adapt(self, x):
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        self.std_ = np.std(x, axis=0, keepdims=True)

    def call(self, inputs):
        eps = tf.keras.backend.epsilon() # small smoothing term
        return (inputs - self.mean_) / (self.std_ + eps)
    
#---------------------------------Discretization Layer------------------------------------
    
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.]) # Categories split to <18, b/t 18 and 50, and >50
age_cat = discretize_layer(age)
# print(age_cat)

discretize_layer = tf.keras.layers.Discretization(num_bins=3) # Now, categories are split into the 33rd and 66th percentiles
discretize_layer.adapt(age)
age_cat = discretize_layer(age)
# print(age_cat)

#---------------------------------Category Encoding Layer-----------------------------------

onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
onehot_layer(age_cat)

# For multihot encoding
two_age_cat = np.array([[1, 0], [2, 2], [2, 0]])
onehot_layer(two_age_cat)

# We can encode both features separately
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3 + 3)
onehot_layer(two_age_cat + [0, 3]) # adds 3 to the second feature

#----------------------------------String Lookup Layer-----------------------------------------

cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
# print(str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])) # ([1], [3], [3], [0])

str_lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot") # one hot encoding
str_lookup_layer.adapt(cities)

str_lookup_layer = tf.keras.layers.StringLookup(num_oov_indices=5)
str_lookup_layer.adapt(cities)
# print(str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]])) # ([[5], [7], [4], [3], [4]])

#----------------------------------Hashing Layer-----------------------------------------------

hashing_layer = tf.keras.layers.Hashing(num_bins=10)
# print(hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]])) # ([[0], [1], [9], [1]])

#-----------------------------------Trainable Embeddings----------------------------------------

tf.random.set_seed(42)
embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2)
# print(embedding_layer(np.array([2, 4, 2])))

tf.random.set_seed(42)
ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(ocean_prox)
lookup_and_embed = tf.keras.Sequential([
    str_lookup_layer,
    tf.keras.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(), output_dim=2)
])
# print(lookup_and_embed(np.array([["<1H OCEAN"], ["ISLAND"], ["<1H OCEAN"]])))

def lookup_and_embed_model():
    X_train_num, X_train_cat, y_train = [...]  # load the training set
    X_valid_num, X_valid_cat, y_valid = [...]  # and the validation set

    num_input = tf.keras.layers.Input(shape=[8], name="num")
    cat_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="cat")
    cat_embeddings = lookup_and_embed(cat_input)
    encoded_inputs = tf.keras.layers.concatenate([num_input, cat_embeddings])
    outputs = tf.keras.layers.Dense(1)(encoded_inputs)
    model = tf.keras.models.Model(inputs=[num_input, cat_input], outputs=[outputs])
    model.compile(loss="mse", optimizer="sgd")
    history = model.fit((X_train_num, X_train_cat), y_train, epochs=5,
                        validation_data=((X_valid_num, X_valid_cat), y_valid))
    
#-----------------------------------Text Preprocessing------------------------------------------
    
train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tf.keras.layers.TextVectorization()
text_vec_layer.adapt(train_data)
# print(text_vec_layer(["Be good!", "Question: be or be?"])) # ([[2, 1, 0, 0],
                                                               # [6, 2, 1, 2]])

text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
# print(text_vec_layer(["Be good!", "Question: be or be?"]))

#--------------------------------Using Pretrained Language Model Components--------------------------

import tensorflow_hub as hub

hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer(tf.constant("To be", "Not to be"))
# print(sentence_embeddings.numpy().round(2))

#---------------------------------------Image Preprocessing Layers------------------------------------

from sklearn.datasets import load_sample_images

images = load_sample_images()["images"]
crop_image_layer = tf.keras.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)

'''---------------------------------------The TensorFlow Datasets Project---------------------------------------------'''

import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

for batch in mnist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]
# Or
mnist_train = mnist_train.shuffle(buffer_size=10_000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)

train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[90%]", "train[90%]", "test"],
    as_supervised=True
)
train_set = train_set.shuffle(buffer_size=10_000, seed=42).batch(32).prefetch(1)
valid_set = valid_set.batch(32).cache()
test_set = test_set.batch(32).cache()
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
