from abc import ABC, abstractmethod
import onnx

class Partitioner(ABC):
    def __init__(self, num_parts: int, overlap: int):
        self.num_parts = num_parts
        self.overlap = overlap

    @abstractmethod
    def split(self, model: onnx.ModelProto) -> list[onnx.ModelProto]:
        """1つのONNXモデルを受け取り、分割された複数のモデルを返す"""
        pass