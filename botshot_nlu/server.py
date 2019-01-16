import falcon
import json

from botshot_nlu.config import ParseHelper


class ParseResource:

    def __init__(self, model_path):
        self.parser = ParseHelper.load(model_path)

    def on_get(self, req, resp):
        text = req.get_param('text', True)
        lang = req.get_param('lang', default="en_US")

        data = self.parser.parse(text)
        resp.body = json.dumps(data)
        resp.status = falcon.HTTP_200


def api(model_dir='model/'):
    api = falcon.API()
    api.add_route("/parse", ParseResource(model_dir))
    return api
