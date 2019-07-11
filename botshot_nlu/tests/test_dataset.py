import pytest
from botshot_nlu import loader


class TestDataset:
    dataset = [
        "hello how are you",
        "i am going to [place:pilsen](plzen)",
        "where is [place](brno)",
        "how do i find [place](zurich hbf)",
        "how do i get to [place](prague)",
        "it is in [place](brno)",
        "i visited [place](prague) yesterday",
        "[place](prague) visitor",
        "how is life in [place](tirano)"
    ]

    def test_read_entities(self):
        tokens, entities = loader._get_entities(self.dataset[1])
        assert tokens == ['i', 'am', 'going', 'to', 'plzen']
        assert entities == ['O', 'O', 'O', 'O', 'B-place']
        tokens, entities = loader._get_entities(self.dataset[3])
        assert tokens == ['how', 'do', 'i', 'find', 'zurich', 'hbf']
        assert entities == ['O', 'O', 'O', 'O', 'B-place', 'I-place']

    def test_get_text_dataset(self):
        dataset = loader._get_text_dataset([(i, None) for i in self.dataset])
        assert len(dataset[0]) == len(dataset[1]) == len(dataset[2])
        dataset = loader._get_text_dataset([(i, None) for i in self.dataset], as_tuple=False)
        assert len(dataset) == len(self.dataset)
        assert dataset[0]['tokens'] == ['hello', 'how', 'are', 'you']
        assert dataset[0]['intent'] is None
