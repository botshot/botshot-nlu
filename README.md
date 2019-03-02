# Botshot NLU

A NLU library for chatbots.

## Training
To train your model, you will need a `nlu_config.yml` file and some training examples.
```
python -m botshot_nlu.cli train --config examples/dataset-2/nlu-config.yml --model-path model
```

## Parsing from CLI
To test your NLU model, you can run:
```
python -m botshot_nlu.cli parse --model-path model
```
Then you can pass sentences from stdin.
```
hello world
{'intent': [{'value': ['"greeting"'], 'confidence': 0.97691810131073}]}
cyan
{'intent': [{'value': [None], 'confidence': 0.7010939121246338}], 'color': [{'value': 'cyan', 'similar_to': 'blue', 'confidence': 0.7178704203026286}]}
red apples
{'intent': [{'value': [None], 'confidence': 0.7010939121246338}], 'color': [{'value': 'red', 'similar_to': 'red', 'confidence': 1.0}], 'fruit': [{'value': 'apples', 'similar_to': 'apple', 'confidence': 0.8046471605767275}]}
```

## Running as a server
You will need a web server like `gunicorn`.
```
gunicorn 'botshot_nlu.server.api("model_dir")'
```
Then you can call for example `http://localhost:8000/parse?text=Hello world`.

## License
(c) Matus Zilinec 2019. All rights reserved.
