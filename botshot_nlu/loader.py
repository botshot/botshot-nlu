import re

import yaml


def _process_example(text, results=None):
    if results is None:
        results = {}
    matches = re.finditer(r"<(!)?([A-Za-z0-9_:]+)>(.*)</\2>", text)
    #text = re.sub(r"(?:[^\\]|^)<(!)?([A-Za-z0-9_:]+)>(.*)</\2?>", r"\1", text)
    # TODO: match predefined entity without content

    entities = []

    for match in matches:
        is_context, label, value = match.groups()
        s, e = match.start(), match.end()
        if s <= 0 or text[s-1] == '\'':
            continue

        entities.append({
            "is_context": bool(is_context),
            "entity": label,
            "value": value,
            "start": s,
            "end": e,
        })

    entities.sort(key=lambda x: x['start'])
    new_text = ""
    off = 0
    for entity in entities:
        s, e = entity['start'], entity['end']
        new_text += text[off:s]
        entity['start'] = len(new_text)
        new_text += entity['value']
        entity['end'] = len(new_text)
        off = e
        if not entity['value']:  # predefined entity
            del entity['value']
    new_text += text[off:]

    results['entities'] = entities
    results['text'] = new_text
    return results


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
        for entity in obj.get("entities", []):
            if entity.get('entity') == 'intent':
                intent = entity['value']
                break
        else:
            continue
        pairs.append((obj.get("text"), intent))
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
