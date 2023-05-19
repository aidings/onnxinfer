# onnxinfer

> onnx inference

### install
`pip install git+https://github.com/aidings/onnxinfer.git`

### modules

#### augment
```python
from onnxinfer import Normalize, Resize, ToTensor, Compose, CenterCrop
# input image(pillow, opencv) 
```

#### math
```python
softmax(input, axis=1)
topk(input, k, axis=None, ascending=False)
```

#### OnnxModel
```python
from onnxinfer import OnnxModel
model = OnnxModel(path, device='cpu|gpu')
result = model({'image': "./image_path"})
```

