# onnx-registry
Intelligent Component Registry web service for managing and using snn, dnn, and ml models, which is stored in onnx format.

registration, download, View Information including visualization tool Netron of onnx format model

Return Inference result through RESTful API

```
$ python onnx_registry.py
```
onnx_registry.py : file upload, download, view onnx model, turn on the onnx inference server by button.

In visualization page, client can download each .onnx model file stored in server. Also can see visualized model offer by NETRON.

# Install & Run
```
$ pip3 install -r requirements.txt
$ python3 onnx_registry.py
```
