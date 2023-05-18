import abc


class ABCBatch(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self):
        self.chunk_id = -1
        self.size = None
        pass

    @abc.abstractclassmethod
    def is_dry(self):
        """
            output: bool, is there another chunk?
        """
        pass

    @abc.abstractclassmethod
    def get_chunk(self, chunk_size=None):
        """
            output: (X, y) or [X, y]
        """
        pass
