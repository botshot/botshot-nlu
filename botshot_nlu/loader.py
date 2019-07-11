import os
import re
import bisect

import yaml


def _process_example(text, results=None, use_html_tags=False):
    if results is None:
        results = {}

    if use_html_tags:
        # <!place>abcd</place>
        matches = re.finditer(r"<(!)?([A-Za-z0-9_:]+)>(.*)</\2>", text)
    else:
        # [!foo:bar](abcd)
        matches = re.finditer(r"\[(!)?(A-Za-z0-9_:)\]\((.*)\)", text)
    #text = re.sub(r"(?:[^\\]|^)<(!)?([A-Za-z0-9_:]+)>(.*)</\2?>", r"\1", text)
    # TODO: match predefined entity without content

    entities = []

    for match in matches:
        # is_context: unused
        is_context, entity, value = match.groups()
        s, e = match.start(), match.end()
        if s <= 0 or text[s-1] == '\'':
            continue

        if ':' in entity:
            # input text != value, as in (place:prague)[praha]
            value = entity.split(':', maxsplit=1)[1]

        entities.append({
            "is_context": bool(is_context),
            "entity": entity,
            "value": value,
            "start": s,
            "end": e,
        })

    entities.sort(key=lambda x: x['start'])
    new_text = ""
    off = 0
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity['start'] = len(new_text)
        entity['end'] = len(new_text)
        new_text += text[off:start]
        if entity['value']:
            new_text += entity['value'].lower()
        else:  # predefined entity
            del entity['value']
        off = end

    new_text += text[off:]

    results['entities'] = entities
    results['text'] = new_text
    return results


def _get_entities(text):
    # find offsets of all spaces
    token_starts = [match.start(1) for match in re.finditer("(?:^|\s+)(\S+)", text)]
    # initialize all entities to "Outside"
    entities = ['O'] * len(token_starts)
    # iterate over entity descriptors like [name:label?](value)
    for match in re.finditer("\[(\S+)\]\(([^)]+)\)", text):
        start, end = match.start(), match.end()
        # obtain entity name and value
        if ':' in match.group(1):
            entity_name, entity_value = match.group(1).split(":", maxsplit=1)
        else:
            entity_name, entity_value = match.group(1), match.group(2)
        # obtain token range of the entity
        first_token = bisect.bisect_left(token_starts, start)
        last_token = bisect.bisect_left(token_starts, end) - 1
        # set entity labels of corresponding tokens to this entity (in BIO encoding)
        entities[first_token] = 'B-' + entity_name.lower()
        if last_token != first_token:
            entities[first_token+1:last_token+1] = ['I-' + entity_name.lower()] * (last_token - first_token)
    # delete entity descriptors from text
    line = re.sub("\[(\S+)\]\(([^\)]+)\)", "\\2", text)
    # tokenize on whitespace (this is necessary in order to match token and entity indices)
    tokens = re.split("\s+", line)
    return tokens, entities
    # TODO: move to pipeline
    features = []
    for token in tokens:
        z_ = []
        z_.append(1.0 if token[0].isupper() else 0.0)
        z_.append(1.0 if token.isalnum() else 0.0)
        features.append(z_)
    X.append(tokens)
    y.append(entities)
    z.append(features)
    print(tokens)
    print(entities)


def read_datasets(*files):

    X, y, z = [], [], []

    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext in ['.yaml', '.yml', '.json']:
            raise NotImplemented()
        else:
            tokens, intents, entities = read_text_file(filename)
            X += tokens
            y += intents
            z += entities

    return X, y, z


def read_text_file(filename: str):
    if not os.path.isfile(filename):
        raise FileNotFoundError()
    # read the file into a list of (text, intent) tuples
    examples = _read_sentences_file(filename)
    # parse the lines into a list of feature dicts
    dataset = _get_text_dataset(examples)
    return dataset


def _get_text_dataset(examples: list, as_tuple=True):
    if as_tuple:
        # tuples: output (tokens, intent, entities)
        dataset = ([], [], [])
    else:
        # dict: output [{line1}, {line2}, ...]
        dataset = []
    for example, intent in examples:
        tokens, entities = _get_entities(example)
        if as_tuple:
            dataset[0].append(tokens)
            dataset[1].append(intent)
            dataset[2].append(entities)
        else:
            example_dict = {"tokens": tokens, "entities": entities, "intent": intent}
            dataset.append(example_dict)
    print(dataset[0])
    return dataset


def _read_sentences_file(filename):
    intent_map = {}

    with open(filename) as fp:
        text = ""
        current_intent = None

        for line in fp:
            split_comments = re.split("(?:[^\\\\]|^)#", line)  # strip comments from line
            line = split_comments[0].strip()

            if re.match("@[^ ]+", line):  # intent specification in beginning of line
                matches = line.split(" ", maxsplit=1)
                intent = matches[0][1:]
                if len(matches) > 1 and matches[1].strip():
                    line = matches[1].strip()
                else:
                    current_intent = intent
                    continue
            else:                         # no intent specification, use previous intent
                if current_intent is None:
                    raise Exception("In file %s: first line must specify intent" % filename)
                intent = current_intent

            if not line.strip():    # empty line
                continue
            elif line[-1] == '\\':  # text continues in next line
                text += line[:-1]
            else:  # line has ended, add it to training examples
                text += line
                intent_map.setdefault(intent, []).append(text.replace("\\", ""))
                text = ""

        if text:
            raise Exception("Unfinished line in file %s" % filename)

    return [(example, intent) for (intent, examples) in intent_map.items() for example in examples]


def _load_sentences(filename):
    out = []
    examples = _read_sentences_file(filename)
    for example, intent in examples:
        processed = _process_example(example)
        processed['entities'].append({"entity": "intent", "value": intent})
        out.append(processed)
    return out


def _load_from_yml(filename):
    with open(filename) as fp:
        obj = yaml.safe_load(fp)
    data = obj.get("data", {})
    for obj in data:
        if 'text' not in obj or 'entities' not in obj:
            raise Exception("Invalid training example {} in file {}".format(obj, filename))
    return data


def load_training_examples(*filenames):
    data = []
    for filename in filenames:
        if filename.endswith(".json") or filename.endswith(".yml") or filename.endswith(".yaml"):
            data += _load_from_yml(filename)  # json examples (wit.ai format)
        else:
            data += _load_sentences(filename)  # text - sentences with markup
    #data = [(obj['text'], intent) for obj, intent in data]
    return data


def as_intent_pairs(data) -> list:
    pairs = []
    for obj in data:
        if not obj.get('text'):
            continue
        intent = None
        entities = []
        for entity in obj.get("entities", []):
            if entity.get('entity') == 'intent':
                intent = entity['value']
                # TODO: what if there are more intents?
            else:
                entities.append(entity)
        if not intent:
            continue  # TODO: what if there are just entities?
        pairs.append((obj.get("text"), intent, entities))
    return pairs


def as_entity_keywords(data) -> dict:
    entities = {}
    for obj in data:
        text = obj.get('text')
        if not text: continue
        for entity_obj in obj.get("entities", []):
            entity = entity_obj.get('entity')
            if entity == 'intent': continue
            value = entity_obj.get('value')
            start = entity_obj.get('start', 0)
            end = entity_obj.get('end', len(text))
            entities.setdefault(entity, {}).setdefault(value, []).append(text[start:end])
    
    for entity, items in entities.items():
        entities[entity] = [{label: expressions} for label, expressions in items.items()]
    return entities
