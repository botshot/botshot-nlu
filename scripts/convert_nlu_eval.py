'''
This script will convert the datasets from NLU Evaluation Corpora (https://github.com/sebischair/NLU-Evaluation-Corpora) to the correct format.
'''

import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", required=True)
parser.add_argument("--target_file", required=True)
args = parser.parse_args()
corpus_file = args.corpus_file
target_file = args.target_file

with open(corpus_file) as fp:
    corpus = json.load(fp)

print("Converting corpus %s" % corpus['name'])

data_training = []
data_testing = []

for sent in corpus['sentences']:
    new_sent = {"text": sent['text'], "entities": []}
    if 'intent' in sent:
        new_sent['entities'].append({"entity": "intent", "value": sent['intent']})
    for entity in sent['entities']:
        new_sent['entities'].append({
            "entity": entity['entity'],
            "start": entity['start'],
            "end": entity['stop'],
            "value": entity['text']
        })
    if sent.get('training'):
        data_training.append(new_sent)
    else:
        data_testing.append(new_sent)

with open(target_file + ".train.json", "w") as fp:
    json.dump({"data": data_training}, fp)

with open(target_file + ".test.json", "w") as fp:
    json.dump({"data": data_testing}, fp)
