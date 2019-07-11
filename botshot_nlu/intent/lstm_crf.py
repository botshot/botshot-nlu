import json
import os
import numpy as np

import tensorflow as tf
from tensorflow.saved_model import simple_save, tag_constants

from botshot_nlu.dataset.intent import IntentDataset, BatchMode
from botshot_nlu.intent import IntentModel, Metrics
from botshot_nlu.pipeline import labels
from botshot_nlu.pipeline.labels import entities_to_array, pad_entities


class LSTMCRFModel(IntentModel):
    # TODO
    MAX_TOKENS = 20
    use_crf = False
    train_embedding = False

    def __init__(self, pipeline, config=None):
        super().__init__(pipeline, config)
        self.intent_bin = labels.LabelBinarizer()
        self.entity_bin = labels.LabelBinarizer()

    def _load(self, feature_dim, label_cnt, entity_cnt):
        if feature_dim is None or feature_dim <= 0:
            raise Exception("Feature dimension should be a positive integer")
        if label_cnt is None or label_cnt <= 0:
            raise Exception("Label count should be a positive integer")

        self.placeholder_x = tf.placeholder(tf.float32, [None, self.MAX_TOKENS, feature_dim], "x")
        self.placeholder_y = tf.placeholder(tf.int32, [None, ], "y")
        self.placeholder_e = tf.placeholder(tf.int32, [None, self.MAX_TOKENS, ], "e")
        self.dropout_rate = tf.placeholder_with_default(1.0, (), "dropout")
        self.threshold = tf.placeholder_with_default(self.config.get("threshold", 0.85), (), "threshold")

        data = self.placeholder_x
        # n_layers = self.config.get("hidden_layers", 3)

        if self.train_embedding:
            data = tf.unstack(data, axis=1)
            embedded = []
            W_emb = tf.get_variable("W_emb", [feature_dim, 300])  # AKA, feature_dim is vocab_len in this case
            b_emb = tf.get_variable("b_emb", [300])
            for i, v in enumerate(data):
                v = tf.add(tf.matmul(v, W_emb), b_emb)
                embedded.append(v)

            data = tf.stack(embedded, axis=1)

        cell_fw = tf.nn.rnn_cell.LSTMCell(
            num_units=32,
            activation='tanh',
            name="lstm_fw_1"
        )
        cell_bw = tf.nn.rnn_cell.LSTMCell(
            num_units=32,
            activation='tanh',
            name="lstm_bw_1"
        )
        # TODO: update, apply dropout, support more layers
        rnn_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, data, dtype=tf.float32)

        if self.use_crf:
            # TODO: use CRF
            raise NotImplementedError()
        else:
            rnn_outputs = tf.concat(rnn_outputs, axis=-1, name="concat_fw_bw")
            position_vectors = tf.unstack(rnn_outputs, axis=1)
            entity_logits = []
            W_pos = tf.get_variable("W_pos", [64, entity_cnt+1])
            # b_pos = tf.get_variable TODO: NAH
            for i, v in enumerate(position_vectors):
                logits = tf.matmul(v, W_pos, name="entity_logits_%d" % i)# + b_pos
                # logits = tf.layers.dense(inputs=v, units=entity_cnt+1, activation=None, name="entity_logits_%d" % i)
                entity_logits.append(logits)

            self.entity_logits = tf.stack(entity_logits, axis=1, name="entity_logits")
            self.entity_probs = tf.nn.softmax(self.entity_logits, axis=2, name="entity_probs")

        final = entity_logits[-1]
        self.intent_logits = tf.layers.dense(
            inputs=final,#tf.concat([state.c for state in final_states], axis=1),
            units=label_cnt,
            activation='relu',
            name="intent_logits_1",
        )
        self.intent_logits = tf.layers.dense(
            inputs=self.intent_logits,  # final_states[0].h,#tf.concat([state.c for state in final_states], axis=1),
            units=label_cnt,
            activation=None,
            name="intent_logits",
        )
        self.intent_probs = tf.nn.softmax(self.intent_logits, axis=-1, name="intent_probs")

        mask = tf.greater_equal(self.intent_probs, self.threshold)
        self.masked_output = tf.multiply(self.intent_probs, tf.cast(mask, tf.float32))
        self.predicted_labels = tf.argmax(self.masked_output, axis=1)
        self.predicted_nomask = tf.argmax(self.intent_probs, axis=1)

        # entity_mask = tf.greater_equal(self.entity_probs, self.threshold)
        # self.masked_entities = tf.multiply(self.entity_probs, tf.cast(entity_mask, tf.float32))
        # self.predicted_entities = tf.argmax(self.masked_entities, axis=2)
        self.predicted_entities = tf.argmax(self.entity_probs, axis=2)   # mask makes no sense, we have the 'O' token

    def _load_training(self, batch_size):
        self.weights = tf.placeholder_with_default([1.] * batch_size, [None], name="weights")
        intent_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.placeholder_y,
            logits=self.intent_logits,
            weights=self.weights,
        )
        entity_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.placeholder_e,
            logits=self.entity_logits,
        ) # TODO: compensate for 'O' tokens?

        self.loss_fn = intent_loss + entity_loss  # TODO: mean or whatnot

        # self.training_op = tf.train.MomentumOptimizer(
        #     learning_rate=self.config.get("learning_rate", 1e-4),
        #     momentum=0.99
        # ).minimize(self.loss_fn)
        self.training_op = tf.train.RMSPropOptimizer(
            learning_rate=1e-3, momentum=0.1, centered=True
        ).minimize(self.loss_fn)
        self.accuracy_op = tf.metrics.accuracy(labels=self.placeholder_y, predictions=self.predicted_labels)
        # TODO: this wont't say much, unbalanced
        self.entity_accuracy_op = tf.metrics.accuracy(labels=self.placeholder_e, predictions=self.predicted_entities)

    def train(self, dataset: IntentDataset) -> Metrics:
        self.unload()

        self.pipeline.fit(*dataset.get_all())  # TODO: why here?
        self.intent_bin.fit(dataset.labels)
        self.entity_bin.fit(dataset.entity_set() | set(['O']))

        session = self._get_session()

        batch_size = self.config.get("batch_size", 16)
        max_steps = self.config.get("max_steps", 100000)
        loss_early_stopping = self.config.get("loss_early_stopping", 0.05)
        dropout = self.config.get("dropout", 0.5)

        with session.graph.as_default():
            self._load(self.pipeline.feature_dim(), dataset.label_count(), dataset.entity_count())  # "O" \in entity_cnt
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
            y = self.intent_bin.encode_labels(y)
            e = self.entity_bin.encode_labels(pad_entities(z))

            # e = self.entity_bin.encode_labels(entities_to_array([item['entities'] for item in y_dict], max_len=self.MAX_TOKENS))

            # print(x, y, e)
            loss, _ = session.run([self.loss_fn, self.training_op], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y: y,
                self.placeholder_e: e,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })

            yy = session.run([self.intent_logits], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y: y,
                self.placeholder_e: e,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })
            #print(yy)

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
                y = self.intent_bin.encode_labels(y)
                e = self.entity_bin.encode_labels(pad_entities(z))

                # TODO: be careful about not fitting unseen entities!

                loss, accuracy = session.run([self.loss_fn, self.accuracy_op], feed_dict={
                    self.placeholder_x: x,
                    self.placeholder_y: y,
                    self.placeholder_e: e,
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
        # TODO: save label binarizers and pipeline

        model_dir = os.path.join(path, "intent")
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, filename))
            os.removedirs(model_dir)

        # os.makedirs(model_dir, exist_ok=True)

        x_info = tf.saved_model.utils.build_tensor_info(self.placeholder_x)
        y_info = tf.saved_model.utils.build_tensor_info(self.predicted_labels)
        p_info = tf.saved_model.utils.build_tensor_info(self.intent_probs)

        e_info = tf.saved_model.utils.build_tensor_info(self.entity_probs)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_info},
            outputs={'y': y_info, 'p': p_info, 'e': e_info},
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
        bin_data = {"intent": self.intent_bin.save(), "entity": self.entity_bin.save()}
        with open(bin_path, "w") as fp:
            json.dump(bin_data, fp)

    def load(self, path: str):

        model_dir = os.path.join(path, "intent")

        bin_path = os.path.join(model_dir, "labels.json")
        with open(bin_path, "r") as fp:
            bin_data = json.load(fp)
        self.intent_bin.load(bin_data["intent"])
        self.entity_bin.load(bin_data["entity"])

        session = self._get_session()
        graph_def = tf.saved_model.loader.load(session, [tag_constants.SERVING], model_dir)
        signature = graph_def.signature_def

        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        x_name = signature[signature_key].inputs['x'].name
        y_name = signature[signature_key].outputs['y'].name
        p_name = signature[signature_key].outputs['p'].name
        e_name = signature[signature_key].outputs['e'].name

        self.placeholder_x = session.graph.get_tensor_by_name(x_name)
        self.predicted_labels = session.graph.get_tensor_by_name(y_name)
        self.intent_probs = session.graph.get_tensor_by_name(p_name)
        self.entity_probs = session.graph.get_tensor_by_name(e_name)

    def predict(self, input):
        input, _, _ = self.pipeline.transform([input])
        probs = self.session.run(self.intent_probs, feed_dict={self.placeholder_x: input})
        prob, label = np.max(probs), np.argmax(probs)

        label = self.intent_bin.decode_labels([label])[0]
        return {"intent": [{"value": label, "confidence": float(prob)}]}
