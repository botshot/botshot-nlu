#!/usr/bin/env python3

import argparse
import os, json, yaml


entity_dicts = {}


def get_expressions(name, base_path):
    expressions = {}
    files = os.listdir(os.path.join(base_path, "expressions"))
    for filename in files:
        with open(os.path.join(base_path, "expressions", filename)) as fp:
            data = json.load(fp)['data']
            for item in data:
                for entity in item['entities']:
                    if entity['entity'] == name:
                        label = entity['value'][1:-1]  # format: \"foo\"
                        expressions.setdefault(label, []).append(item['text'])
                        break
    expressions = [ {"value": k, "samples": v} for k, v in expressions.items() ]
    return expressions


def main(args):
    base_path = args.path
    entity_files = os.listdir(os.path.join(base_path, "entities"))
    for entity_file in entity_files:
        with open(os.path.join(base_path, "entities", entity_file)) as fp:
            data = json.load(fp)
            if data['data']['builtin']:
                print("Skipping builtin %s" % data['data']['name'])
                continue
            lookups = data['data']['lookups']
            strategy = "bow" if "trait" in lookups else "keywords"
            name = data['data']['name']
            
            if strategy == "keywords":
                expressions = []
                for obj in data['data']['values']:
                    expressions.append({
                        "label": obj['value'],
                        "samples": obj['expressions'],
                    })
            else:
                expressions = get_expressions(name, base_path)

            entity_dicts[name] = {"strategy": strategy, "data": expressions}
    if args.out:
        if os.path.exists(args.out):
            print("Out path %s already exists, please clean it up." % args.out)
            exit(1)
        os.makedirs(os.path.join(args.out, "nlu"))
        for entity, data in entity_dicts.items():
            with open(os.path.join(args.out, "nlu", entity + ".yml"), "w") as fp:
                yaml.dump(data, fp)
        # TODO: generate nlu-config.yml
    else:
        for entity, data in entity_dicts.items():
            print("#@@@ Entity %s @@@#" % entity)
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("path", help="Path to Wit export root folder.", type=str)
    argp.add_argument("--out", help="Where to write the output.", type=str)
    args = argp.parse_args()
    main(args)
