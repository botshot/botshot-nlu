import falcon


class ParseResource:

    def __init__(self):
        pass

    def on_get(self, req, resp):
        text = req.get_param('text', True)
        lang = req.get_param('lang', default="en_US")

