import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.applications import efficientnet

"""
Training Parameters
"""

seed = 111
np.random.seed(seed)  # reseed the RandomState instance
tf.random.set_seed(seed)  # set the global seed

IMAGES_PATH = "Flicker8k_Dataset/"
IMAGE_SIZE = (299, 299)  # Desired image size
VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 25  # Fixed length allowed for any sequence
MIN_SEQ_LENGTH = 5  # Fixed length allowed for any sequence
EMBED_DIM = 512  # Dimension for image and token embeddings
FF_DIM = 512  # Per-layer unit in the feed-forward network
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

"""
Preparing the Dataset
"""


def load_captions_data(filename):
    """
    Loads caption data and maps to corresponding image
    """
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            img_name, caption = line.split("\t")  # name and caption separated by tab
            img_name = img_name.split("#")[0]
            img_name = os.path.join(IMAGES_PATH, img_name.strip())

            tokens = caption.strip().split()

            if len(tokens) < MIN_SEQ_LENGTH or len(tokens) > MAX_SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """
    Split the captioning dataset into train and validation sets
    """
    all_images = list(caption_data.keys())  # Get all image names

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(
        len(caption_data) * train_size
    )  # Split into training and validation sets. 80% training by default.

    training_data = {
        img_name: caption_data[img_name]
        for img_name in all_images[:train_size]  # Sets first 80% as training data
    }

    validation_data = {
        img_name: caption_data[img_name]
        for img_name in all_images[train_size:]  # Sets last 20% as validation data
    }

    return training_data, validation_data


"""
Load the dataset and split into training and validation sets
"""

captions_mapping, text_data = load_captions_data("Flickr8k_text/Flickr8k.token.txt")
train_data, valid_data = train_val_split(captions_mapping)
print(f"Number of training samples: {len(train_data)}")  # 6114
print(f"Number of validation samples: {len(valid_data)}")  # 1529

"""
Vectorizing the text data using the TextVectorization layer
"""


def custom_sandardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQ_LENGTH,
    standardize=custom_sandardization,
)
vectorization.adapt(text_data)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)

"""
Building a tf.data.Dataset pipeline for training. Generate pairs of images and corresponding captions
"""


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))
print(f"Training dataset: {train_dataset}")
print(f"Validation dataset: {valid_dataset}")

"""
Building the model
"""


def get_cnn_model():
    """
    A CNN used to extract the image features
    """
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    # Freeze our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_moddel = keras.models.Model(base_model.input, base_model_out)
    return cnn_moddel


class TransformerEncoderBlock(layers.Layer):
    """
    TransformerEncoderBlock Class

    This class is a building block used in transformer-based neural networks for processing and transforming input data.

    Args:
        embed_dim (int): Dimensionality of the input data.
        dense_dim (int): Dimensionality of intermediate (hidden) layers.
        num_heads (int): Number of attention heads to use.
        **kwargs: Additional keyword arguments that can be passed.

    Methods:
        call(inputs, training, mask=None):
            Process the input data.

            Args:
                inputs: The input data to be processed by the encoder block.
                training (bool): A boolean indicating whether the network is in training mode.
                mask (Optional): An additional mask for the attention mechanism. Default is None.

            Returns:
                Tensor: The transformed input data.

            Note:
                This class is an essential part of a Transformer-based neural network, and it helps in capturing complex patterns in the input data through self-attention and feedforward neural networks.
    """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Attention model
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        """
        Process the input data.

        Args:
            inputs: The input data to be processed by the encoder block.
            training: A boolean indicating whether the network is in training mode.
            mask: (Optional) An additional mask for the attention mechanism. Default is None.

        Returns:
            Tensor: The transformed input data.
        """
        # Apply layer normalization to the input data.
        inputs = self.layernorm_1(inputs)
        # Pass the data through a feedforward neural network layer.
        inputs = self.dense_1(inputs)

        # Use multi-head self-attention to process the input data.
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )

        # Apply another layer of normalization and
        # add the result of the attention mechanism to the original input.
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(layers.Layer):
    """
    PositionalEmbedding Class

    This class is responsible for generating positional embeddings to be added to token embeddings in a Transformer-based model.

    Args:
        sequence_length (int): The length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the embeddings.
        **kwargs: Additional keyword arguments that can be passed.

    Methods:
        call(inputs):
            Generate positional embeddings and add them to token embeddings.

            Args:
                inputs: The input data for which positional embeddings are generated.

            Returns:
                Tensor: The input data with positional embeddings added.

        compute_mask(inputs, mask=None):
            Compute a mask based on input data.

            Args:
                inputs: The input data for which the mask is computed.
                mask: (Optional) An additional mask. Default is None.

            Returns:
                Tensor: A mask indicating which elements in the input data are not equal to zero.

            Note:
                This class is typically used as a building block in Transformer models to provide information about the position of tokens within a sequence.
    """

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        """
        Generate positional embeddings and add them to token embeddings.

        Args:
            inputs: The input data for which positional embeddings are generated.

        Returns:
            Tensor: The input data with positional embeddings added.
        """
        length = tf.shape(inputs)[-1]
        # Generate an array of positions to represent positions of tokens
        positions = tf.range(start=0, limit=length, delta=1)
        # Compute embeddings for input tokens
        embedded_tokens = self.token_embeddings(inputs)
        # Scale by embed_scale
        embedded_tokens = embedded_tokens * self.embed_scale
        # Generate positional embeddings for calculated positions
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens * embedded_positions

    def compute_mask(self, inputs, mask=None):
        """
        Compute a mask based on input data.

        Args:
            inputs: The input data for which the mask is computed.
            mask: (Optional) An additional mask. Default is None.

        Returns:
            Tensor: A mask indicating which elements in the input data are not equal to zero.
        """
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """
    TransformerDecoderBlock Class

    This class represents a single block within a Transformer decoder. It is responsible for processing and transforming decoder inputs.

    Args:
        embed_dim (int): The dimension of the embedding.
        ff_dim (int): The dimension of the feed-forward network.
        num_heads (int): The number of attention heads.
        **kwargs: Additional keyword arguments that can be passed.

    Methods:
        call(inputs, encoder_outputs, training, mask=None):
            Perform decoding and transformation of input data.

            Args:
                inputs: The decoder input data.
                encoder_outputs: The outputs from the encoder.
                training: A boolean indicating whether the model is in training mode.
                mask: (Optional) An additional mask. Default is None.

            Returns:
                Tensor: Predicted outputs.

        get_causal_attention_mask(inputs):
            Generate a causal attention mask.

            Args:
                inputs: The input data for which the mask is generated.

            Returns:
                Tensor: The causal attention mask.

            Note:
                This class is used to create one block in the Transformer decoder architecture.

    """

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        # Create layers for self-attention and feed-forward
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        # Create layer normalizations
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        # Create embedding layer for positional encoding
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=MAX_SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )

        # Create an output layer
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        # Create dropout layers
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)

        # Enable masking
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        """
        Perform decoding and transformation of input data.

        Args:
            inputs: The decoder input data.
            encoder_outputs: The outputs from the encoder.
            training: A boolean indicating whether the model is in training mode.
            mask: (Optional) An additional mask. Default is None.

        Returns:
            Tensor: Predicted outputs.
        """
        # Apply positional embeddings to input data
        inputs = self.embedding(inputs)

        # Generate causal attention mask
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            # Prepare padding mask
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)

            # Combine causal and padding masks
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        # Perform self-attention with first layer
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )

        # Apply first layer normalization
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Self-attention with second layer
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )

        # Second layer normalization
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Apply feed-forward neural network
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)

        # Generate predictions
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        """
        Generate a causal attention mask.

        Args:
            inputs: The input data for which the mask is generated.

        Returns:
            Tensor: The causal attention mask.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # Get image embedding
        img_embed = self.cnn_model(batch_img)

        # Pass each of the 5 captions to decoder
        # with encoder outputs and compute loss & accuracy
        # for each caption
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # Get list of all trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # Get gradients
            grads = tape.gradient(loss, train_vars)

            # Update trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # Update trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # Get image embeddings
        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation
)


"""
Model Training
"""

# Loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


# Learning Rate Scheduler for optimizer
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )


# Create Learning Rate Scheduler
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

# Compile model
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = MAX_SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    sample_img = np.random.choice(valid_images)

    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    encoded_img = caption_model.encoder(img, training=False)

    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace("<end>", "").strip()
    print("Predicted Caption: ", decoded_caption)


generate_caption()
generate_caption()
generate_caption()
