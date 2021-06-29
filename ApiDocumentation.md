# ONNX Inference

### quick guide

## 1. model 폴더에 파일 저장
HTTP
```
curl -X POST -F "model=@{test_file}" http://0.0.0.0:5065/model/{name}
```
## URI Parameters

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>test_file<code>  | test_data 폴더에 저장될 파일  | string |


 
# 2. Reference
### HTTP
```
POST http://0.0.0.0:5065/
```
### Request Body Parameter
(raw/json)
 
 1. jpg
```
{
  "model_name":"model/{model_name}",
  "test_image":"test_data/{test_image}"
}
```
  2. npy
```
{
  "model_name":"model/{model_name}",
  "test_file":"test_data/{test_file}"
}
```

## Request Body Parameters

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>model_name<code> | model 폴더 내의 모델명     | string |
| <code>test_image<code> | test_data 폴더 내의 파일명 | string |
| <code>test_file<code>  | test_data 폴더 내의 파일명 | string |
 
<br><br><br>

 # Reference Example
 ### HTTP
```
POST http://0.0.0.0:5065/
```
### Request Body Parameter
(raw/json)
```
{
  "model_name":"model/vgg_onnx.onnx",
  "test_image":"test_data/test_mnist_9.jpg"
}
```
### Sample Response
```
{
  "result": "9"
}
```
