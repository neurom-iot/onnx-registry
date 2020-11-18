import onnx
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
# 분류다 평가지표 분류 평가지표
from ONNX_Registry_ver1_2.onnx_to_nengo_model import toNengoModel, classification_accuracy, classification_error, objective

import nengo
import nengo_dl
import tensorflow as tf

""" 
onnx 파일이 
skl2onnx로 -> https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md
keras2onnx로 -> https://github.com/onnx/onnx/blob/master/docs/Operators.md
tensorflow2onnx 로 만들어 졌을 때를 가정
"""

# 모델 판별 Module 클래스
# 주요기능 1. 해당 모델이 어떤 프레임워크로 만들어졌는지
# 주요기능 2. 해당 모델이 어떤 연산자 기본인지 -> ai.onnx / onnx.ml /
class Distinguish_onnx:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_framework = None
        self.testX = None; # TEST DATA
        self.onnx_load_model = onnx.load_model(model_path)
        self.model_type = None

    # model type 이 snn 이지 아닌지 구분
    # 어떠한 프레임워크로 만들어 졌는가.
    def getModelFramework(self):
        # onnx 만든 프레임워크
        # model type 이 snn 이지 아닌지 구분
        if self.model_type == None:
            self.model_type = self.onnx_load_model.producer_name
            self.model_type = self.model_type.replace('2onnx','')
            print(self.model_type)
            self.model_framework = self.model_type
            if self.model_type == 'skl':
                self.model_framework = 'scikit-learn'
                return 'ML ' + self.model_framework
            elif self.model_type == 'keras':
                self.model_framework = 'keras'
                return 'DL ' + self.model_framework
            elif self.model_type =='tf':
                self.model_framework = 'tensorflow'
                return 'DL ' + self.model_framework
        elif self.model_type == 'snn':
            print('model_type 이 snn 입니다')
            self.model_framework = 'nengo'
            return 'SNN' + self.model_framework

        else: # 오류 발생시
            raise ValueError('사용되어진 Framework를 알수 없습니다')

    # 어떠한 연산자를 기본으로 사용하고 있는가
    def getModelOperator(self):
        for i in range(len(self.onnx_load_model.graph.node)):
            op_type = self.onnx_load_model.graph.node[i].op_type.lower()
            if op_type == "lif" or op_type == "lifrate" or op_type == "adaptivelif" \
                    or op_type == "adaptivelifrate" or op_type == "izhikevich" \
                    or op_type == "softlifrate":
                self.model_type = 'snn'
                print(self.model_type)
                return
        return

    #dl과 ml의 구분_"ai.onnx"
    def getModeldomain(self):
        model_domain_opertor = self.onnx_load_model.domain
        return model_domain_opertor

    # ONNX Runtime 으로 추론 - ml, dl
    def ort_run(self, testX):
        self.testX = testX
        sess = ort.InferenceSession(self.model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: self.testX.astype(np.float32)})[0]
        print('-- 추론 Complete -- ')

        return pred_onx

    def nengo_run(self, testX):
        self.testX = testX
        print('mnist data 준비')

        # to nengo 로 한거지
        otn = toNengoModel(self.model_path)
        model = otn.get_model()
        inp = otn.get_inputProbe()
        pre_layer = otn.get_endLayer()

        # 돌리는 것
        with model:
            out_p = nengo.Probe(pre_layer)
            out_p_filt = nengo.Probe(pre_layer, synapse=0.01)

        # ----------------------------------------------------------- run
        sim = nengo_dl.Simulator(model, device="/cpu:0")

        # when testing our network with spiking neurons we will need to run it
        # over time, so we repeat the input/target data for a number of
        # timesteps.

        n_steps = 30
        print(self.testX.shape) # 30, 28, 28, 1
        self.testX = self.testX.reshape((self.testX.shape[0], -1))
        print(self.testX.shape) # 30, 784
        test_images = np.tile(self.testX[:, None, :], (1, n_steps, 1))
        print(test_images.shape)

        # load parameters
        print('load_params')
        sim.load_params("weights/mnist_params_adam_0.001_3_100")

        sim.compile(loss={out_p_filt: classification_accuracy})
        data = sim.predict(test_images)
        sim.close()
        print('simulator 종료')
        return data
