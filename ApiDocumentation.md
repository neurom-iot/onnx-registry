# ONNX Inference REST API Reference


## 1.(1) 모델 업로드
- 모델 파일을 서버에 업로드 한다.

HTTP
```
POST http://SERVER_ADDR/model/upload/{name}
```
### URI Parameters

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>name<code>  | 서버의 모델 폴더에 저장될 이름 | string |


### Request Body
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>model_file<code>  | 업로드할 모델 파일  | string |

### Response
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| 200 OK  | 모델 업로드 성공  | string |
  
## 1.(2) 모델 업로드
- 모델 파일을 서버에 업로드 하고, json파일을 읽어 결과를 리턴받는다.

HTTP
```
POST http://SERVER_ADDR/model/upload/{name}
```
### URI Parameters

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>name<code>  | 서버의 모델 폴더에 저장될 이름 | string |


### Request Body
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>model_file<code>  | 업로드할 모델 파일  | string |
| <code>jsonf<code>       | 읽혀질 json 파일  | string |

### Response
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| 200 OK  | 모델 업로드 성공  | string |


## 2. 모델을 이용한 추론
- 입력 데이터를 주고 서버에 저장된 모델을 이용하여 추론하고 결과를 리턴 받는다.

HTTP
```
POST http://SERVER_ADDR/model/predict/{name}
```

### URI Parameters

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>name<code>  | 사용할 모델 이름 | string |


### Request Body
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| <code>input_data<code>  | 모델의 입력 데이터  | string |

### Response
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| 200 OK  | 추론 성공  | string |

### Response Body
- json

|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------:|
| result  | 추론 결과  | string |

## 3. 모델 다운로드

## 4. 모델 사용 통계

# Example

## 모델 업로드하고 추론하기
### 모델 업로드
```
curl -X POST -F "model_file=@{knn_iris.onnx}" http://SERVER_ADDR/model/upload/knn_iris
``` 
- URI 파라미터
  - name = knn_iris
- Body 파라미터
  - model_file = knn_iris.onnx 파일 컨텐츠 
- Response
  ```
  200 OK
  ```
  
## 모델 업로드하고 json 파일 읽기
### 모델 업로드
```
curl -X POST -F "model_file=@{knn_iris.onnx}" -F "jsonf=@{ex.json}" http://SERVER_ADDR/model/upload/knn_iris
``` 
- URI 파라미터
  - name = knn_iris
- Body 파라미터
  - model_file = knn_iris.onnx 파일 컨텐츠 
  - jsonf = ex.json 파일
- Response
  ```
  200 OK
  ```
  
### 모델 추론
```
curl -X POST -F "input_data=@{input.jpg}" http://SERVER_ADDR/model/predict/knn_iris
```
- URI 파라미터
  - name = knn_iris
- Body 파라미터
  - input_data = input.jpg 파일 컨텐츠
- Response
  ```
  200 OK
  ```
- Response Body
  ```
  {
    "result": "9"
  }
  ```

