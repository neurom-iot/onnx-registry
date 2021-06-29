# ONNX Inference

 ### quick guide

model 폴더에 저장
```
curl -X POST -F "model=@{test_data}" http://0.0.0.0:5065/model/{model_name}
```

추론결과 

## url
```
http://0.0.0.0:5065/
```
## request body 
(raw/json)
```
{
  "model_name":"model/vgg_onnx.onnx",
  "test_image":"test_data/test_mnist_9.jpg"
}
```

## Example Response
```
{
  "result": "9"
}
```


## Request Body

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>model_name<code> | model 폴더 내의 파일     | string |
| <code>test_image<code> | test_data 폴더 내의 파일 | string |
| <code>test_file<code>  | test_data 폴더 내의 파일 | string |
  
 
## Response
|         Status         |       Description       |
|------------------------|-------------------------|
| 200 OK | result of a successful operation. |
 
