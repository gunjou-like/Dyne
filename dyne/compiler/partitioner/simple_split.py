import copy
import onnx
from .base import Partitioner

class SimpleSplitter(Partitioner):
    """
    1次元CNNモデルを単純に空間分割するSplitter。
    Dynamic Axesが設定されていることを前提とし、
    重みのスライスは行わず、入出力シェイプの定義のみを書き換える。
    """

    def split(self, model: onnx.ModelProto) -> list[onnx.ModelProto]:
        print(f"Splitting model into {self.num_parts} parts with overlap {self.overlap}...")
        
        # MVPでは「全幅=100」と仮定して計算します。
        TOTAL_WIDTH = 100  
        
        base_width = TOTAL_WIDTH // self.num_parts
        sub_models = []

        for i in range(self.num_parts):
            part_model = copy.deepcopy(model)
            
            # Part 0: [0, 50] + overlap(right)
            # Part 1: [50, 100] + overlap(left)
            start_idx = max(0, (i * base_width) - self.overlap)
            end_idx = min(TOTAL_WIDTH, ((i + 1) * base_width) + self.overlap)
            
            part_width = end_idx - start_idx
            print(f"  Part {i}: Grid range [{start_idx} : {end_idx}] -> Input Width: {part_width}")

            # 3. ONNXグラフのInput/Outputシェイプ定義を書き換える
            self._update_tensor_dim(part_model.graph.input, dim_index=2, new_size=part_width)
            self._update_tensor_dim(part_model.graph.output, dim_index=2, new_size=part_width)

            self._update_tensor_dim(part_model.graph.input, dim_index=0, new_size=1)
            self._update_tensor_dim(part_model.graph.output, dim_index=0, new_size=1)
            
            part_model.graph.name = f"{model.graph.name}_part_{i}"
            sub_models.append(part_model)

        return sub_models

    def _update_tensor_dim(self, tensors, dim_index, new_size):
        """指定されたテンソルリストの指定次元のサイズを書き換える"""
        for tensor in tensors:
            # 修正箇所 1: tensor_typeを持っているか確認
            if not tensor.type.HasField("tensor_type"):
                continue

            # 修正箇所 2: shapeフィールドを持っているか確認 (has_shape -> HasField("shape"))
            if not tensor.type.tensor_type.HasField("shape"):
                continue
            
            # 次元インデックスが範囲内か安全確認
            if len(tensor.type.tensor_type.shape.dim) <= dim_index:
                continue

            # 次元取得
            dim = tensor.type.tensor_type.shape.dim[dim_index]
            
            # 修正箇所 3: dim_param (文字列) があれば消去し、dim_value (数値) をセット
            if dim.HasField("dim_param"):
                dim.ClearField("dim_param")
            
            dim.dim_value = new_size