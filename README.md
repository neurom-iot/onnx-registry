# onnx-registry
Intelligent Component Registry web service for managing and using snn, dnn, and ml models, which is stored in onnx format.

registration, download, View Information including visualization tool Netron of onnx format model

Return Inference result through RESTful API

### 디렉토리 구조
```
├── ONNX-registry
|   ├── log_folder
|   |   └── log.txt
|   ├── model         
|   ├── test_data
|   ├── templates
|   |   ├── mainpage.html                  
|   |   ├── netron_wrapper.html              
|   |   └── onnx_manager.html             
|   ├── onnx_distinguish_run.py
|   ├── onnx_inference_restapi.py    #Inference Server 
|   ├── onnx_registry.py             #onnx registry web by FLASK
|   ├── onnx_to_nengo_model.py  
|   ├── requirements.txt
```

<br/>


### onnx_registry.py
파일 업로드, 모델 다운로드 및 모델 목록 확인

버튼을 통해 onnx 추론 서버 on

![registry](https://user-images.githubusercontent.com/71939195/121996644-ee0c8400-cde3-11eb-8926-f9da46122ee5.PNG)






시각화 페이지에서 각 onnx 모델에 대한 정보를 볼 수 있고 서버에 저장된 모델을 다운

NETRON으로 모델 시각화

![visualized](https://user-images.githubusercontent.com/71939195/121996925-72f79d80-cde4-11eb-8146-441215924664.PNG)

<br/>

### onnx_inference_restapi.py
FLASK RESTful api를 통한 추론 결과 반환

![dosik](https://user-images.githubusercontent.com/71939195/121997231-fca76b00-cde4-11eb-9b65-2af22c5cbfa7.PNG)

추론 결과 반환 방법
 - REST API curl command.txt 파일의 curl command를 통한 추론 결과 반환
 - postman agent 
 
 ![postman](https://user-images.githubusercontent.com/71939195/121997720-c1596c00-cde5-11eb-81f8-758d082cd702.png)
 
 -> postman agent를 통한 iris 데이터 추론 결과 반환 예시


# Install & Run
```
$ pip3 install -r requirements.txt
$ python3 onnx_registry.py
```
