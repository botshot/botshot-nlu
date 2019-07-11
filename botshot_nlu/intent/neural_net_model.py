import json
import os
from abc import abstractmethod, ABC

import numpy as np

import tensorflow as tf
from sklearn import metrics
from tensorflow.saved_model import simple_save, tag_constants

from botshot_nlu.dataset.intent import IntentDataset, BatchMode
from botshot_nlu.intent import IntentModel, Metrics
from botshot_nlu.pipeline import labels


class TensorflowModel(IntentModel, ABC):

    def _get_session(self):
        if not hasattr(self, 'session') or self.session is None:
            print("Creating a new Tensorflow session")
            graph = tf.Graph()
            self.session = tf.Session(graph=graph)
        return self.session

    @abstractmethod
    def _load(self, feature_dim, label_cnt):
        pass

    @abstractmethod
    def _load_training(self, batch_size):
        pass

    def train(self, dataset: IntentDataset) -> Metrics:
        self.unload()
        session = self._get_session()

        batch_size = self.config.get("batch_size", 16)
        max_steps = self.config.get("max_steps", 100000)
        loss_early_stopping = self.config.get("loss_early_stopping", 0.02)
        dropout = self.config.get("dropout", 0.5)

        with session.graph.as_default():
            self._load(self.pipeline.feature_dim(), dataset.label_count())
            self._load_training(batch_size)
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
            self.saver = tf.train.Saver()

        if dataset.count() < batch_size:
            raise Exception(
                "There are not enough training examples. "
                "Please set your batch_size lower than %d"
                % dataset.count()
            )

        #        logical_epoch = dataset.count() % batch_size
        dataset.set_mode(BatchMode.BALANCED)
        losses = []

        for step in range(1, max_steps + 1):  # TODO: accuracy as >=threshold instead of argmax

            x, y, z = dataset.get_batch(batch_size)
            x, y, z = self.pipeline.transform(x, y, z)
            y = self.label_bin.encode_labels(y)

            loss, _ = session.run([self.loss_fn, self.training_op], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y: y,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })

            losses.insert(0, loss)
            losses = losses[:100]
            if len(losses) == 100 and sum(losses) / len(losses) <= loss_early_stopping:
                print("Early stopping due to loss")
                break

            print("Step %d/%d: loss %f" % (step, max_steps, loss))

        return self.test(dataset)

    def unload(self):
        if hasattr(self, 'session') and self.session is not None:
            with self.session.as_default():
                tf.reset_default_graph()
            self.session.close()
            self.session = None
            # TODO: remove variables from class


class NeuralNetModel(IntentModel):

    def __init__(self, pipeline, config=None):
        super().__init__(pipeline, config)

    def _load(self, feature_dim, label_cnt):
        if feature_dim is None or feature_dim <= 0:
            raise Exception("Feature dimension should be a positive integer")
        if label_cnt is None or label_cnt <= 0:
            raise Exception("Label count should be a positive integer")

        self.placeholder_x = tf.placeholder(tf.float32, [None, feature_dim], "x")
        self.placeholder_y = tf.placeholder(tf.int32, [None, ], "y")
        self.dropout_rate = tf.placeholder_with_default(1.0, (), "dropout")
        self.threshold = tf.placeholder_with_default(self.config.get("threshold", 0.85), (), "threshold")

        data = self.placeholder_x
        n_layers = self.config.get("hidden_layers", 3)

        for i in range(n_layers):
            data = tf.layers.dense(
                inputs=data,
                units=self.config.get("num_units", 64),
                activation=tf.nn.tanh,
                name="dense_%d" % i,
            )
            data = tf.layers.dropout(inputs=data, rate=self.dropout_rate)

        self.logits = tf.layers.dense(
            inputs=data,
            units=label_cnt,
            activation=None,
            name="logits",
        )
        self.probabilities = tf.nn.softmax(self.logits, axis=-1)

        mask = tf.greater_equal(self.probabilities, self.threshold)
        self.masked_output = tf.multiply(self.probabilities, tf.cast(mask, tf.float32))
        self.predicted_labels = tf.argmax(self.masked_output, axis=1)
        self.predicted_nomask = tf.argmax(self.probabilities, axis=1)

    def _load_training(self, batch_size):
        self.weights = tf.placeholder_with_default([1.] * batch_size, [None], name="weights")
        self.loss_fn = tf.losses.sparse_softmax_cross_entropy(
            labels=self.placeholder_y,
            logits=self.logits,
            weights=self.weights,
        )
        self.training_op = tf.train.MomentumOptimizer(
            learning_rate=self.config.get("learning_rate", 0.003),
            momentum=0.99
        ).minimize(self.loss_fn)
        self.accuracy_op = tf.metrics.accuracy(labels=self.placeholder_y, predictions=self.predicted_labels)

    def train(self, dataset: IntentDataset) -> Metrics:
        self.unload()
        self.label_bin = labels.LabelBinarizer()
        self.pipeline.fit(*dataset.get_all())
        self.label_bin.fit(dataset.labels)

        session = self._get_session()

        batch_size = self.config.get("batch_size", 16)        
        max_steps = self.config.get("max_steps", 100000)
        loss_early_stopping = self.config.get("loss_early_stopping", 0.02)
        dropout = self.config.get("dropout", 0.5)

        with session.graph.as_default():
            self._load(self.pipeline.feature_dim(), dataset.label_count())
            self._load_training(batch_size)

            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
            self.saver = tf.train.Saver()

        if dataset.count() < batch_size:
            raise Exception(
                "There are not enough training examples. "
                "Please set your batch_size lower than %d"
                % dataset.count()
            )

#        logical_epoch = dataset.count() % batch_size
        dataset.set_mode(BatchMode.BALANCED)
        losses = []

        for step in range(1, max_steps+1):  # TODO: accuracy as >=threshold instead of argmax

            x, y, z = dataset.get_batch(batch_size)
            x, y, z = self.pipeline.transform(x, y, z)
            y = self.label_bin.encode_labels(y)

            loss, _ = session.run([self.loss_fn, self.training_op], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y: y,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })

            losses.insert(0, loss)
            losses = losses[:100]
            if len(losses) == 100 and sum(losses) / len(losses) <= loss_early_stopping:
                print("Early stopping due to loss")
                break

            print("Step %d/%d: loss %f" % (step, max_steps, loss))

        return self.test(dataset)

    def test(self, dataset: IntentDataset):
        batch_size = 16
        i = 0
        avg_loss = 0.
        accuracy = 0.
        dataset.set_mode(BatchMode.SEQUENTIAL)
        session = self._get_session()

        while True:
            try:
                x, y, z = dataset.get_batch(batch_size)
                x, y, z = self.pipeline.transform(x, y, z)
                y = self.label_bin.encode_labels(y)

                loss, accuracy = session.run([self.loss_fn, self.accuracy_op], feed_dict={
                    self.placeholder_x: x,
                    self.placeholder_y: y,
                })
                avg_loss = (avg_loss * i + loss) / (i + 1)
                # accuracy is already a moving average
                i += 1
            except IntentDataset.EndOfEpoch:
                break
        print("Loss: %f, Accuracy: %f" % (avg_loss, accuracy[1]))
        return Metrics(avg_loss, accuracy[1], -1, -1, -1)

    def _get_session(self):
        if not hasattr(self, 'session') or self.session is None:
            print("Creating a new Tensorflow session")
            graph = tf.Graph()
            self.session = tf.Session(graph=graph)
        return self.session

    def unload(self):
        if hasattr(self, 'session') and self.session is not None:
            with self.session.as_default():
                tf.reset_default_graph()
            self.session.close()
            self.session = None
            # TODO: remove variables from class

    def save(self, path: str):

        if not hasattr(self, 'session') or self.session is None:
            raise Exception("Session is None, have you called train() ?")
        
        super().save(path)
        model_dir = os.path.join(path, "intent")
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, filename))
            os.removedirs(model_dir)
        #os.makedirs(model_dir, exist_ok=True)

        x_info = tf.saved_model.utils.build_tensor_info(self.placeholder_x)
        y_info = tf.saved_model.utils.build_tensor_info(self.predicted_labels)
        p_info = tf.saved_model.utils.build_tensor_info(self.probabilities)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_info},
            outputs={'y': y_info, 'p': p_info},
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
        bin_data = {"intent": self.label_bin.save()}
        with open(bin_path, "w") as fp:
            json.dump(bin_data, fp)

    def load(self, path: str):
        model_dir = os.path.join(path, "intent")

        bin_path = os.path.join(model_dir, "labels.json")
        with open(bin_path, "r") as fp:
            bin_data = json.load(fp)
        self.label_bin = labels.LabelBinarizer()
        self.label_bin.load(bin_data["intent"])

        session = self._get_session()
        graph_def = tf.saved_model.loader.load(session, [tag_constants.SERVING], model_dir)
        signature = graph_def.signature_def
        
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        x_name = signature[signature_key].inputs['x'].name
        y_name = signature[signature_key].outputs['y'].name
        p_name = signature[signature_key].outputs['p'].name

        self.placeholder_x = session.graph.get_tensor_by_name(x_name)
        self.predicted_labels = session.graph.get_tensor_by_name(y_name)
        self.probabilities = session.graph.get_tensor_by_name(p_name)

    def predict(self, input):
        input, _, _ = self.pipeline.transform([input])
        probs = self.session.run(self.probabilities, feed_dict={self.placeholder_x: input})
        prob, label = np.max(probs), np.argmax(probs)

        label = self.label_bin.decode_labels([label])[0]
        return {"intent": [{"value": label, "confidence": float(prob)}]}
