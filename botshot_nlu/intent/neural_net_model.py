import tensorflow as tf


from botshot_nlu.dataset.intent import IntentDataset, BatchMode
from botshot_nlu.intent import IntentModel, Metrics


class NeuralNetModel(IntentModel):

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
        self.pipeline.fit(*dataset.get_all())
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

            x, y = dataset.get_batch(batch_size)
            x, y = self.pipeline.transform(x, y)

            loss, _ = session.run([self.loss_fn, self.training_op], feed_dict={
                self.placeholder_x: x,
                self.placeholder_y: y,
                # weights: batch_weights,
                self.dropout_rate: dropout,
            })

            losses.insert(0, loss)
            losses = losses[:10]
            if sum(losses) / len(losses) <= loss_early_stopping:
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
                x, y = dataset.get_batch(batch_size)
                x, y = self.pipeline.transform(x, y)
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
        super().save(path)
        # TODO
