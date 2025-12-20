import onnx

def load_model(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model