# n3ml-onnx

## How to use


# onnx-registry
- 지능형 컴포넌트 레지스트리

### directory structure
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


## onnx_registry.py
파일 업로드,다운로드 및 모델 목록 확인

![new_ui](https://user-images.githubusercontent.com/71939195/136668065-3a9de036-9222-4081-a402-43df438b4988.png)


각 컴포넌트의 package.json에서 논리적 컴포넌트 이름, 식별자, 키워드, 컴포넌트 생성 저자등의 정보 parsing

![detail](https://user-images.githubusercontent.com/71939195/136668243-cfcefb79-cabd-41e2-b5e9-868f30f11b01.png)



## Install & Run
```
$ pip3 install -r requirements.txt
$ python3 onnx_registry.py
```
