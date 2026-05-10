from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, *args, **kwargs):
        ...
