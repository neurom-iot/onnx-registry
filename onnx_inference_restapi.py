import os
import numpy as np

from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename

from onnx_distinguish_run import Distinguish_onnx
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_FOLDER = 'model'
INPUT_FOLDER = 'input'

app = Flask('ONNX')
api = Api(app)


# Model 종류 판별해주는 함수
def Distinguish_Model(model_path):
    distinguish_onnx = Distinguish_onnx(model_path=model_path)

    distinguish_onnx.model_path = model_path
    # snn 인지 아닌지 먼저 판별
    # 1. op_type 확인 - snn 활성화 함수 (lif, .. 가 들어있으면 snn 으로 정함)
    distinguish_onnx.getModelOperator()
    print('--- 사용되어진 Model은 ', model_path, '입니다')

    # 2. 사용된 딥러닝 Framework
    model_framework = distinguish_onnx.getModelFramework()
    print('--- 사용되어진 Model Framework는 ', model_framework, '입니다')

    return distinguish_onnx, model_framework  # class 정보와, model_framework 정보 return

@app.route("/")
def home():
    return "Hello, Flask!"


@app.route('/model/<name>', methods=['GET']) # curl -X GET http://127.0.0.1:5000/model/test.onnx
def get_model(name):
    # return model statistics
    resp = jsonify({'message' : f'model {name} statistics and information'})
    resp.status_code = 200
    return resp

@app.route('/model/<name>',methods=['POST'])
# curl -X POST -F "model=@test_data/test_mnist_2.jpg" http://121.129.244.69:5065/model/test.onnx
def predict_model(name):
    try:
        model_file = request.files['model']
    except:
        resp = jsonify({'message': 'No input'})
        resp.status_code = 400
        return resp
    filename = secure_filename(model_file.filename)
    MODEL_FOLDER = "temp"

    model_file.save(os.path.join(MODEL_FOLDER, filename))
    # predict output using input filename
    resp = jsonify({'message': f'model: {name}, predict result'})
    resp.status_code = 200
    return resp


@app.route('/model/<name>', methods=['PUT']) # curl -X PUT -F "model=@sangsangbugi-coding.png" http://127.0.0.1:5000/model/test.onnx
def put_model(name):
    # save model
    try:
        model_file = request.files['model']
    except:
        resp = jsonify({'message' : 'No input'})
        resp.status_code = 400
        return resp

    filename = secure_filename(name)
    model_file.save(os.path.join(MODEL_FOLDER, filename))
    resp = jsonify({'message' : f'saving model {filename}, {model_file.filename}'})
    resp.status_code = 200
    return resp


# @app.route('/test', methods=['GET', 'POST'])
class IrisEstimator(Resource):
    def post(self):
        try:
            # 파라미터 파싱
            parser = reqparse.RequestParser()
            parser.add_argument('model_name', required=True, help='model_name')  #

            ## iris ML 용 (방법 2가지)
            # 1번째 (변수 인자 직접 입력)
            # 2번째 test_file 인자로 npy입력 : test-data/iris_X_test.npy
            parser.add_argument('sepal_length', type=float, help='sepal_length is required')
            parser.add_argument('sepal_width', type=float, help='sepal_width is required')
            parser.add_argument('petal_length', type=float, help='petal_length is required')
            parser.add_argument('petal_width', type=float, help='petal_width is required')

            ## mnist DL 용
            parser.add_argument('test_file')
            parser.add_argument('test_image')
            args = parser.parse_args()

            ## 모델 판별 함수
            # distinguish_onnx(클래스 객체), model_framework(어떤 모델인지 반환)
            distinguish_onnx, model_framework = Distinguish_Model(args['model_name'])

            # 위 판별식 결과에 따라서 구분
            # SNN 인경우..
            if model_framework == 'SNNnengo':

                # SNN 방법1 - Mnist 실제 이미지 파일로
                if args['test_image'] != None:
                    # load an image from file
                    image = load_img(args['test_image'], grayscale=True)  # image load
                    image = img_to_array(image)  # convert the image pixels to a numpy array
                    image = image[np.newaxis, :, :, :].astype(np.float32)  # 맨앞에 차원 1 추가(batch size이자 1임)
                    print("image shape:", image.shape)

                    features = image

                # SNN 방법2 - Mnist .npy로
                else:
                    features = np.load(args['test_file'])
                    print(features.shape)
                pred_nengo = distinguish_onnx.nengo_run(features)

                predict_probabilty = np.array(
                    list(pred_nengo.values())[0])  # ordering dictionary 형태라서 out_p 가져올려고 이렇게 함.
                print(predict_probabilty.shape)  # 10, 30, 10 -> 10장이 30초동안 10개의 확률값 (0~9)

                # 가운데 timestemp 중 마지막 timestemp 경우만 뽑기 위해서. (그게 최종)
                predict_probabilty = predict_probabilty[:, predict_probabilty.shape[1] - 1, :10]  # 맨 마지막 timestep 결과
                print(predict_probabilty.shape)  # 10, 10 -> 10장에 대한 0~9까지의 확률값
                predict_argmax = predict_probabilty.argmax(axis=1)  #

                # 최종 결과 REST API 출력을 위해 list를 String으로 변경
                result_list = list(map(str, predict_argmax.flatten()))
                result = ','.join(result_list)
                print(result)
                return jsonify({'result': result})

            # ML, DL 인 경우..
            else:
                model_domain = distinguish_onnx.getModeldomain()
                # ML 인경우
                if model_domain == 'ai.onnx':
                    # ml의 경우1 - test_file : npy로 하는법
                    if args['test_file'] != None:
                        features = np.load(args['test_file'])

                    # ml의 경우2 - args 직접 변수값 줘서 하는법
                    else:
                        features = [args['sepal_length'], args['sepal_width'], args['petal_length'],
                                    args['petal_width']]  # 인자들합침
                        features = np.reshape(features, (1, 4))

                    # list 형태로 일자로 쭉 펼쳐주긔
                    pred_onx = distinguish_onnx.ort_run(features)  # numpy 배열 넘겨줌
                    result_list = list(map(str, pred_onx.flatten()))

                    # iris data 직접 라벨링
                    for r in range(0, len(result_list)):
                        if result_list[r] == '0':
                            result_list[r] = 'setosa'
                        elif result_list[r] == '1':
                            result_list[r] = 'versicolor'
                        elif result_list[r] == '2':
                            result_list[r] = 'virginica'

                    # 최종 결과 REST API 출력을 위해 list를 String으로 변경
                    result = ','.join(result_list)
                    print(result)
                    return jsonify({'result': result})

                # DL 인 경우
                else:
                    # DL 방법1 - Mnist 실제 이미지 파일로
                    if args['test_image'] != None:
                        # load an image from file
                        image = load_img(args['test_image'], grayscale=True)  # image load
                        image = img_to_array(image)  # convert the image pixels to a numpy array
                        image = image[np.newaxis, :, :, :].astype(np.float32)  # 맨앞에 1 추가(batch size이자 1임)
                        print("image shape:", image.shape)
                        features = image


                    # DL 방법2 - Mnist npy로
                    else:
                        features = np.load(args['test_file'])

                    # ONNX Runtime 사용용
                    pred_onx = distinguish_onnx.ort_run(
                        features)  # numpy 배열 넘겨주기 (10, 10) 임 -> 10개 즉 0~9 까지의 mnist 각각 확률이 쭉 펼쳐져있음 소수점으로
                    pred_onx = pred_onx.argmax(axis=1)  # 가장 큰거 선택
                    print(pred_onx, pred_onx.shape)

                    # 최종 결과 REST API 출력을 위해 list를 String으로 변경
                    result_list = list(map(str, pred_onx.flatten()))
                    result = ','.join(result_list)
                    print(result)
                    return jsonify({'result': result})  # 최종 결과

        except Exception as e:
            return {'error2': str(e)}



api.add_resource(IrisEstimator, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5065, debug=True)  # 5750 포트로 실행

## curl 명령어
# ML 방법1 - iris 인자 직접입력
# curl - d "sepal_length=6.3&sepal_width=3.3&petal_length=6.0&petal_width=2.5&model_name=model/logreg_iris.onnx" - X GET http: // localhost: 5095
#
# # ML 방법2 - .npy
# curl - d "model_name=model/logreg_iris.onnx&test_file=test_data/iris_X_test.npy" - X GET http: // localhost: 5095
#
# # DL 방법1 - image
# curl - d "model_name=model/lenet-1.onnx&test_image=test_data/test_mnist_9.jpg" - X GET http: // localhost: 5095
#
# # DL 방법2 - .npy
# curl - d "model_name=model/lenet-1.onnx&test_file=test_data/mnist_X_test_10.npy" - X GET http: // localhost: 5095
#
# # SNN 방법1 - .npy
# curl - d "model_name=model/lenet-1_snn.onnx&test_file=test_data/mnist_X_test_10.npy" - X GET http: // localhost: 5095
#
# # SNN 방법2 - image
# curl - d "model_name=model/lenet-1_snn.onnx&test_image=test_data/test_mnist_5.jpg" - X GET http: // localhost: 5095
######
