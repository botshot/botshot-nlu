import json
import os
import random

import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from botshot_nlu.dataset.intent import IntentDataset, BatchMode
from botshot_nlu.intent import IntentModel, Metrics
from botshot_nlu.intent.neural_net_model import TensorflowModel
from botshot_nlu.pipeline import labels


class TrainedEmbedding(TensorflowModel):

    def train(self, dataset: IntentDataset) -> Metrics:
        # initialize pipeline and label binarizer
        self.label_bin = labels.LabelBinarizer()
        self.label_bin.fit(dataset.labels)
        self.pipeline.fit(*dataset.get_all())

        batch_size = self.config.get("batch_size", 32)
        dataset.set_mode(BatchMode.BALANCED)
        if dataset.count() < batch_size:
            raise Exception(
                "There are not enough training examples. "
                "Please set your batch_size lower than %d"
                % dataset.count()
            )

        session = self._get_session()
        with session.graph.as_default():
            self.init_model(self.pipeline.feature_dim(), dataset.label_count())
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

        self.training_loop(dataset)
        return self.test(dataset)

    def init_model(self, feature_dim, label_cnt):
        self.placeholder_x = tf.placeholder(tf.float32, [None, feature_dim], "x")
        self.placeholder_y_true = tf.placeholder(tf.int32, [None, ], "y_true")
        self.placeholder_y_neg = tf.placeholder(tf.int32, [None, 5], "y_neg")
        self.dropout_rate = tf.placeholder_with_default(1.0, (), "dropout")

        embedding_dim = self.config.get("embedding_dim", 300)

        x = tf.layers.dense(self.placeholder_x, embedding_dim, activation='relu')
        x = tf.layers.dropout(x, self.dropout_rate)
        x = tf.layers.dense(x, embedding_dim, activation='relu')
        self.x = tf.layers.dense(x, embedding_dim)
        self.x_embedding = tf.nn.l2_normalize(self.x, axis=-1)

        y_true = tf.one_hot(self.placeholder_y_true, depth=label_cnt)
        y_neg = tf.one_hot(self.placeholder_y_neg, depth=label_cnt)

        y_layers = [
            tf.layers.Dense(embedding_dim, activation='relu'),
            tf.layers.Dropout(self.dropout_rate),
            tf.layers.Dense(embedding_dim, activation='relu'),
            tf.layers.Dense(embedding_dim),
        ]

        for l in y_layers:
            y_true = l(y_true)
        self.y_embedding = tf.nn.l2_normalize(y_true, axis=-1)
        # pos_distance = tf.reduce_sum(tf.multiply(self.x_norm, self.y_embedding))
        # pos_distance = tf.losses.cosine_distance(self.x_norm, self.y_embedding, axis=-1)
        self.pos_distance = tf.reduce_sum(self.x_embedding * self.y_embedding, axis=-1)
        print(self.x_embedding.shape)

        y_neg = tf.unstack(y_neg, axis=1)
        neg_distances = []
        for v in y_neg:
            for l in y_layers:
                v = l(v)
            v_norm = tf.nn.l2_normalize(v, axis=-1)
            # dist = tf.reduce_sum(self.x_embedding * v_norm, axis=-1)
            dist = tf.reduce_sum(self.x_embedding * v_norm, axis=-1)
            neg_distances.append(dist)
        # neg_distance = tf.reduce_mean(tf.stack(neg_distances, axis=1))
        neg_distance = tf.stack(neg_distances, axis=1) #tf.reduce_sum(tf.stack(neg_distances))
        print(neg_distance)

        # self.loss_fn = tf.reduce_mean(np.abs(pos_distance)) + tf.reduce_max(tf.exp(-tf.abs(neg_distance)))
        self.loss_fn = tf.maximum(0.8 - self.pos_distance, 0.0) + tf.maximum(0.2 + tf.reduce_max(neg_distance, -1), 0.0)
        self.loss_fn = tf.reduce_mean(self.loss_fn)
        self.training_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss_fn)

    def training_loop(self, dataset):
        # load hyperparameters
        batch_size = self.config.get("batch_size", 32)
        max_steps = self.config.get("max_steps", 1500)
        loss_early_stopping = self.config.get("loss_early_stopping", 0.000)
        dropout = self.config.get("dropout", 0.5)

        losses = []

        # x, y, z = dataset.get_batch(batch_size)
        # x, y, z = self.pipeline.transform(x, y, z)
        # train for max_steps batches
        for step in range(1, max_steps + 1):  # TODO: accuracy as >=threshold instead of argmax
            # preprocess batch
            x, y, z = dataset.get_batch(batch_size)
            x, y, z = self.pipeline.transform(x, y, z)
            # y = self.label_bin.encode_labels(y)

            y_in = np.zeros([batch_size, 5])
            y_true= self.label_bin.encode_labels(y)
            labels = set(dataset.labels) - set(y)
            y_neg= self.label_bin.encode_labels([random.sample(labels, k=5) for i in range(batch_size)])

            # TODO: y_neg minus those in batch
            # run training step
            loss, _ = self.session.run([self.loss_fn, self.training_op], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y_true: y_true,
                self.placeholder_y_neg: y_neg,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })

            losses.insert(0, loss)
            losses = losses[:100]
            if len(losses) == 100 and sum(losses) / len(losses) <= loss_early_stopping:
                print("Early stopping due to loss")
                break

            print("Step %d/%d: loss %f" % (step, max_steps, loss))

    def test(self, dataset: IntentDataset):
        dataset.set_mode(BatchMode.SEQUENTIAL)
        batch_size = self.config.get("batch_size", 32)
        i, avg_loss, accuracy = 0, 0., 0.
        session = self._get_session()
        correct, total = 0., 0.

        while True:
            try:
                x, y, z = dataset.get_batch(32)
                x, y, z = self.pipeline.transform(x, y, z)
                # y = self.label_bin.encode_labels(y)

                y_true = self.label_bin.encode_labels(y)
                y_all = self.label_bin.encode_labels(dataset.labels)

                # y_in = np.zeros([batch_size, 5])
                # y_in[:, 0] = self.label_bin.encode_labels(y)
                # y_in[:, 1:] = self.label_bin.encode_labels([random.sample(dataset.labels, k=4)] * batch_size)

                y_all = session.run(self.y_embedding, feed_dict={self.placeholder_y_true: y_all})
                x = session.run(self.x_embedding, feed_dict={self.placeholder_x: x})
                self.y_all = y_all
                # dists = np.multiply(x, y_all)
                assert x.shape == (32, 300), x.shape
                assert y_all.shape == (39, 300), y_all.shape
                # dists = np.matmul(x, y_all.T)
                for i in range(batch_size):
                    dists = [np.sum(np.multiply(x[i], y_all[j])) for j in range(39)]
                    y_pred = np.argmax(np.abs(dists))
                    if y_pred == y_true[i]:
                        correct += 1
                    else:
                        print(self.label_bin.decode_labels([int(y_pred)]), '!=', self.label_bin.decode_labels([int(y_true[i])]))
                total += batch_size
                i += 1
            except IntentDataset.EndOfEpoch:
                break
        accuracy = correct / total
        print("Loss: %f, Accuracy: %f" % (avg_loss, accuracy))
        return Metrics(avg_loss, accuracy, -1, -1, -1)

    def save(self, path: str):
        if not hasattr(self, 'session') or self.session is None:
            raise Exception("Session is None, have you called train() ?")

        super().save(path)
        model_dir = os.path.join(path, "intent")
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, filename))
            os.removedirs(model_dir)
        # os.makedirs(model_dir, exist_ok=True)

        x_info = tf.saved_model.utils.build_tensor_info(self.placeholder_x)
        e_info = tf.saved_model.utils.build_tensor_info(self.x_embedding)
        # p_info = tf.saved_model.utils.build_tensor_info(self.probabilities)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_info},
            outputs={'e': e_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

        with self.session.graph.as_default():
            builder.add_meta_graph_and_variables(
                self.session, [tag_constants.SERVING],
                {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
                strip_default_attrs=True
            )
        builder.save(as_text=True)

        bin_path = os.path.join(model_dir, "labels.json")
        bin_data = {"intent": self.label_bin.save(), "y_all": self.y_all.tolist()}
        with open(bin_path, "w") as fp:
            json.dump(bin_data, fp)

    def load(self, path: str):
        model_dir = os.path.join(path, "intent")

        bin_path = os.path.join(model_dir, "labels.json")
        with open(bin_path, "r") as fp:
            bin_data = json.load(fp)
        self.label_bin = labels.LabelBinarizer()
        self.label_bin.load(bin_data["intent"])
        self.y_all = bin_data['y_all']

        session = self._get_session()
        graph_def = tf.saved_model.loader.load(session, [tag_constants.SERVING], model_dir)
        signature = graph_def.signature_def

        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        x_name = signature[signature_key].inputs['x'].name
        e_name = signature[signature_key].outputs['e'].name
        # p_name = signature[signature_key].outputs['p'].name

        self.placeholder_x = session.graph.get_tensor_by_name(x_name)
        self.x_embedding = session.graph.get_tensor_by_name(e_name)
        # self.probabilities = session.graph.get_tensor_by_name(p_name)

    def _load(self, feature_dim, label_cnt):
        pass

    def _load_training(self, batch_size):
        pass

    def predict(self, input):
        input, _, _ = self.pipeline.transform([input])

        x = self.session.run(self.x_embedding, feed_dict={self.placeholder_x: input})[0]
        dists = [np.sum(np.multiply(x, self.y_all[j])) for j in range(39)]
        y_pred = np.argmax(np.abs(dists))
        print(self.label_bin.decode_labels([int(y_pred)]))

        # input, _, _ = self.pipeline.transform([input])
        # probs = self.session.run(self.probabilities, feed_dict={self.placeholder_x: input})
        # prob, label = np.max(probs), np.argmax(probs)

        # label = self.label_bin.decode_labels([label])[0]
        # return {"intent": [{"value": label, "confidence": float(prob)}]}

        return {}