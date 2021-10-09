# 지능형 컴포넌트 레지스트리 REST API 문서
- Na-컴포넌트 파일을 업로드/다운로드
- AI 서버의 URL, 입/출력 정보를 업로드/다운로드


## 1. Na-컴포넌트 업로드
- Na-컴포넌트는 .zip으로 압축된 형식

HTTP
```
POST http://SERVER_ADDR/component/upload
```

### Request Body
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| <code>file<code>  | 업로드할 컴포넌트 파일  | string |

### Response
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| 200 OK                 | 업로드 성공             | string |

### 레지스트리에서 처리
- 전송받은 컴포넌트 zip파일을 저장하고
- 압축을 풀어서 그 안에 있는 manifest 내용을 웹 UI에서 보여줌
- manifest 안에 ID가 있음

## 2. Na-컴포넌트 다운로드
- 서버에 있는 컴포넌트 파일(.zip)을 다운로드

HTTP
```
GET http://SERVER_ADDR/component/{ID}
```

### URI Parameters
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| <code>ID<code>  | 다운로드할 컴포넌트 ID  | string |

## 3. AI 서버 URL 등록 
- AI 서버의 URL과 입출력 타입 등을 업로드/다운로드

HTTP
```
POST http://SERVER_ADDR/aiserver/upload
```

### Request Body
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| <code>json<code>  | 업로드할 AI서버 정보 파일  | string |

### Response
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| 200 OK                 | 업로드 성공             | string |

### 레지스트리에서 처리
- 전송받은 json파일을 저장, json 파일에는 해당 서버의 모듈을 사용하기 위한 REST API 정보 포함
- 해당 json파일에 대한 ID 생성
- json 파일 내용
    - URL
    - Reuqest_Body
    - Response_Body
    - Response
    - description
- 웹 UI에서는 URL과 description 등 보여줌


## 4. AI 서버 URL 다운로드
- AI 서버 정보 json파일 다운로드

HTTP
```
GET http://SERVER_ADDR/aiserver/{ID}
```

### URI Parameters
|          Name          |       Description       |  Type |
|------------------------|-------------------------|-------|
| <code>ID<code>  | 다운로드할 json ID | string |