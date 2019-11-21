from botshot_nlu.dataset.intent import IntentDataset
from botshot_nlu.intent import IntentModel, Metrics
from botshot_nlu.pipeline.embedding import Embedding
from botshot_nlu.pipeline import labels
from botshot_nlu.dataset.intent import BatchMode

import os, json, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingSimilarityModel(IntentModel):

    MAX_TOKENS = 20
    FEATURE_DIM = 300
    MIN_CONFIDENCE = 0.3

    def __init__(self, config, pipeline):
        super().__init__(config, pipeline)
        self.pipeline = pipeline
        self.intent_bin = labels.LabelBinarizer()
        # self.entity_bin = labels.LabelBinarizer()

    def train(self, dataset: IntentDataset) -> Metrics:
        self.pipeline.fit(*dataset.get_all())  # TODO: why here?
        self.intent_bin.fit(dataset.labels)
        # self.entity_bin.fit(dataset.entity_set() | set(['O']))

        self._load(self.pipeline.feature_dim(), len(self.intent_bin.i2l))

        max_steps = 1000
        batch_size = 32
        dataset.set_mode(BatchMode.BALANCED)
        losses = []

        loss_fn = nn.CosineEmbeddingLoss()
        optim = torch.optim.Adam(self.parameters(), lr=0.003)

        for step in range(1, max_steps + 1):  # TODO: accuracy as >=threshold instead of argmax
            x, y, z = dataset.get_batch(batch_size)
            # print(x)
            x, y, z = self.pipeline.transform(x, y, z)
            y = self.intent_bin.encode_labels(y)
            # e = self.entity_bin.encode_labels(pad_entities(z))
            choices_set = set(range(0, len(self.intent_bin.i2l)))
            choices_for = [list(choices_set - set([y[i]])) for i in range(batch_size)]
            y_neg = np.array([random.choice(choices_for[i]) for i in range(batch_size)])
            # y_neg[y_neg == y[0]] = 0
            optim.zero_grad()
            x_neg = torch.tensor(x, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            y_neg = torch.tensor(y_neg, dtype=torch.long)

            x_zero = torch.zeros([1, self.MAX_TOKENS], dtype=torch.long)
            y_zero = torch.zeros([1], dtype=torch.long)  # avoids bias

            x = torch.cat((x, x_neg, x_zero), dim=0)
            y = torch.cat((y, y_neg, y_zero), dim=0)
            x_emb = F.tanh(self.word_emb(x))
            x_emb = torch.sum(x_emb, dim=-2)
            y_emb = F.tanh(self.lbl_emb(y))
            # print(x_emb.shape, y_emb.shape)
            sims = [1.0] * batch_size + [-1.0] * (batch_size + 1)
            # print(sims)
            print(F.cosine_similarity(x_emb, y_emb))
            loss = loss_fn(x_emb, y_emb, torch.tensor(sims, dtype=torch.float))
            print("It %d Loss %f" % (step, loss))
            loss.backward()
            optim.step()
        # text = input()
        # print(self.predict(text))

    def parameters(self):
        return [x for x in self.word_emb.parameters()] + [x for x in self.lbl_emb.parameters()]

    def test(self, dataset: IntentDataset) -> Metrics:
        pass

    def save(self, model_dir: str):

        super().save(model_dir)

        bin_path = os.path.join(model_dir, "labels.json")
        bin_data = {"intent": self.intent_bin.save()}#, "entity": self.entity_bin.save()}
        with open(bin_path, "w") as fp:
            json.dump(bin_data, fp)

        torch.save(self.word_emb.state_dict(), os.path.join(model_dir, "word-emb.pt"))
        torch.save(self.lbl_emb.state_dict(), os.path.join(model_dir, "lbl-emb.pt"))


    def load(self, model_dir: str):

        super().load(model_dir)

        bin_path = os.path.join(model_dir, "labels.json")

        with open(bin_path, "r") as fp:
            bin_data = json.load(fp)
        self.intent_bin.load(bin_data["intent"])
        # self.entity_bin.load(bin_data["entity"])

        self._load(self.pipeline.feature_dim(), len(self.intent_bin.i2l))
        self.word_emb.load_state_dict(torch.load(os.path.join(model_dir, "word-emb.pt")))
        self.lbl_emb.load_state_dict(torch.load(os.path.join(model_dir, "lbl-emb.pt")))
        self.word_emb.eval()
        self.lbl_emb.eval()


    def _load(self, vocab_x, vocab_y):
        self.word_emb = nn.Embedding(vocab_x, self.FEATURE_DIM, padding_idx=0)
        self.lbl_emb = nn.Embedding(vocab_y, self.FEATURE_DIM, padding_idx=0)

    def unload(self):
        del self.word_emb
        del self.lbl_emb

    def predict(self, text, **kwargs):
        print([text])
        x, _, _ = self.pipeline.transform([text], None, None)
        # print(x)
        ys = range(len(self.intent_bin.i2l))
        x = torch.tensor([x], dtype=torch.long)
        y = torch.tensor([ys], dtype=torch.long)
        # print(x.shape, y.shape)
        x_emb = self.word_emb(x)
        x_emb = torch.sum(x_emb, dim=-2)
        y_emb = self.lbl_emb(y)
        sim = F.cosine_similarity(F.tanh(x_emb), F.tanh(y_emb), dim=-1)
        # print(sim)
        prob = sim.max().item()
        l_i = sim.argmax().item()
        label = self.intent_bin.i2l[l_i]
        if float(prob) < self.MIN_CONFIDENCE:
            return {}
        return {"intent": [{"value": label, "confidence": float(prob)}]}
