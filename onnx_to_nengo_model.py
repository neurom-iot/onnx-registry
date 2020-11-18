import os
import re
import onnx
import numpy as np
import nengo
import nengo_dl
import tensorflow as tf

class toNengoModel:
    def __init__(self, onnx_path, amplitude=0.01):
        self.nengoCode = ""
        self.amplitude = amplitude  # default amplitude = 0.01
        self.onnx_model = onnx.load(onnx_path) # onnx model
        self.model = None
        self.end_layer = None
        self.set_network() # 모델 network 구성 시작

    # 모델 분석 시작 부분
    def set_network(self):
        # init code generating
        model = self.gen_init()

        # layer code generating
        onnx_model_graph = self.onnx_model.graph  # graph가
        node_len = len(onnx_model_graph.node)
        input_shape = np.array(onnx_model_graph.input[0].type.tensor_type.shape.dim)  # input 데이터에 대한 shape dim 이 나타남 ( 28, 28, 1)

        # temp는 최종적으로 input 크기를 가리킴 = input_shape 로 마지막 지칭
        temp = []
        regex = re.compile(":.*\d*")
        for index in range(1, len(input_shape)):
            data = regex.findall(str(input_shape[index]))[0]
            temp.append(int(data[2:len(data)]))
        input_shape = temp  # temp => [28,28,1] 만들기 위함

        # model 과 input 크기 list 가지고 들어감.
        # model 객체와, output shape와, 만들어지고 있는 x 반환(node객체)
        model, output_shape, pre_layer = self.gen_input(model, input_shape)
        self.model = model

        # node들 갯수만큼 op_type 찾아서 변환
        for index in range(node_len):
            node_info = onnx_model_graph.node[index]  # node 들 정보
            op_type = node_info.op_type.lower()  # op_type 리턴
            if op_type == "conv":
                model, output_shape, pre_layer = self.convert_conv2d(model, pre_layer, output_shape, index, onnx_model_graph)
                self.model = model
            elif op_type == "batchnormalization":
                 model, output_shape, pre_layer = self.convert_batchnormalization2d(model, pre_layer, output_shape, node_info)
                 self.model = model
            elif op_type == "maxpool":
                code, output_shape, pre_layer = self.convert_maxpool2d(model, pre_layer, output_shape, node_info)
                self.model = model
            elif op_type == "averagepool":
                code, output_shape, pre_layer = self.convert_avgpool2d(model, pre_layer, output_shape, node_info)
                self.model = model
            elif op_type == "reshape":
                regex = re.compile("flatten")
                if regex.findall(node_info.name):
                    code, output_shape, pre_layer = self.convert_flatten(model, pre_layer, output_shape)
                    self.model = model
            elif op_type == "matmul":
                code, output_shape, pre_layer = self.convert_dense(model, pre_layer, output_shape, index, onnx_model_graph)
                self.model = model
        self.model = model
        self.end_layer = pre_layer

    # NENGO 모델 기본적인 설정
    def gen_init(self):
        model = nengo.Network()
        with model:
            model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            nengo_dl.configure_settings(trainable=False)
        return model

    # input node 얻기 - Nengo의 Node 객체 얻는 것.
    def gen_input(self, model, input_shape):
        length = len(input_shape)  # 3차원 [28,28,1]
        dim = 1
        for index in range(0, length):  #
            dim *= input_shape[index]  # dim은 전체 28x28x1

        with model:
            self.inp = nengo.Node([0] * dim)  # Nengo Node (28x28x1)개 생성
            x = self.inp
        output_shape = input_shape  # 현재 하나만 했을때 output shape 는 이것

        # model 객체와, output shape와, 만들어지고 있는 x 반환(node객체)
        return model, output_shape, x  # x를 return 하면서 모델을 계속 쌓아감

    def convert_conv2d(self, model, pre_layer, input_shape, index, onnx_model_graph):
        onnx_model_graph_node = onnx_model_graph.node
        node_info = onnx_model_graph_node[index]
        neuron_type = self.get_neuronType(index, onnx_model_graph_node)
        filters = self.get_filterNum(node_info, onnx_model_graph)
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                kernel_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "auto_pad":
                padding = node_info.attribute[index].s.decode('ascii').lower()
                if padding != "valid":
                    padding = "same"
        if padding == "same":
            output_shape = [input_shape[0], input_shape[1], filters]
        else:
            output_shape = [int((input_shape[0] - kernel_size) / strides + 1),
                            int((input_shape[1] - kernel_size) / strides + 1), filters]
        with model:
            x = nengo_dl.Layer(tf.keras.layers.Convolution2D(
                filters=filters, kernel_size=kernel_size, padding=padding))\
                (pre_layer, shape_in=(input_shape[0], input_shape[1], input_shape[2]))

            # activation
            if neuron_type == "lif":
                x = nengo_dl.Layer(nengo.LIF(amplitude=self.amplitude))(x)
                print('activation lif 추가 완료')
            elif neuron_type == "lifrate":
                x = nengo_dl.Layer(nengo.LIFRate(amplitude=self.amplitude))(x)
            elif neuron_type == "adaptivelif":
                x = nengo_dl.Layer(nengo.AdaptiveLIF(amplitude=self.amplitude))(x)
            elif neuron_type == "adaptivelifrate":
                x = nengo_dl.Layer(nengo.AdaptiveLIFRate(amplitude=self.amplitude))(x)
            elif neuron_type == "izhikevich":
                x = nengo_dl.Layer(nengo.Izhikevich(amplitude=self.amplitude))(x)
            elif neuron_type == "softlifrate":
                x = nengo_dl.Layer(nengo_dl.neurons.SoftLIFRate(amplitude=self.amplitude))(x)
            elif neuron_type == None:  # default neuron_type = LIF
                x = nengo_dl.Layer(nengo.LIF(amplitude=self.amplitude))(x)
            print('convert_conv2d finish')
        return model, output_shape, x

    # batchnormalization
    def convert_batchnormalization2d(self, model, pre_layer, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "momentum":
                momentum = round(node_info.attribute[index].f, 4)
                if momentum == 0:
                    momentum = 0.99
            elif node_info.attribute[index].name == "epsilon":
                epsilon = round(node_info.attribute[index].f, 4)
                if epsilon == 0:
                    epsilon = 0.001
        with model:
            x = nengo_dl.Layer(tf.keras.layers.batch_normalization)(
                pre_layer, shape_in=(input_shape[0], input_shape[1], input_shape[2]), momentum=momentum, epsilon=epsilon)
        output_shape = input_shape
        print('convert_batchnormalization2d finish')
        return model, output_shape, x  # x를 return 하면서 모델을 계속 쌓아감

    # maxpooling
    def convert_maxpool2d(self, model, pre_layer, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                pool_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
        output_shape = [int(input_shape[0] / strides), int(input_shape[1] / strides), input_shape[2]]
        with model:
            x = nengo_dl.Layer(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides))(
                pre_layer, shape_in=(input_shape[0], input_shape[1], input_shape[2]))
        print('convert_maxpool2d finish')
        return model, output_shape, x

    # average pooling
    def convert_avgpool2d(self, model, pre_layer, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                pool_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
        output_shape = [int(input_shape[0] / strides), int(input_shape[1] / strides), input_shape[2]]
        with model:
            x = nengo_dl.Layer(tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides))(
                pre_layer, shape_in=(input_shape[0], input_shape[1], input_shape[2]))
        print('convert_avgpool2d finish')
        return model, output_shape, x

    # flatten
    def convert_flatten(self, model, pre_layer, input_shape):
        with model:
            x = nengo_dl.Layer(tf.keras.layers.Flatten)(pre_layer)
        output_shape = 1
        for index in range(len(input_shape)):
            output_shape *= input_shape[index]
        output_shape = [output_shape, 1]
        print('Flatten finish')
        return model, output_shape, x

    # matmul과 같은 존재
    def convert_dense(self, model, pre_layer, input_shape, index, onnx_model_graph):
        onnx_model_graph_node = onnx_model_graph.node
        node_info = onnx_model_graph_node[index]
        dense_num = self.get_dense_num(node_info, onnx_model_graph)
        neuron_type = self.get_neuronType(index, onnx_model_graph_node)  # node들 지나다니면서 - neuron이 op_type이 어떤건지 찾음

        with model:
            x = nengo_dl.Layer(tf.keras.layers.Dense(units=dense_num))(pre_layer)
            if neuron_type != "softmax":
                if neuron_type == "lif":
                    x = nengo_dl.Layer(nengo.LIF(amplitude=self.amplitude))(x)
                elif neuron_type == "lifrate":
                    x = nengo_dl.Layer(nengo.LIFRate(amplitude=self.amplitude))(x)
                elif neuron_type == "adaptivelif":
                    x = nengo_dl.Layer(nengo.AdaptiveLIF(amplitude=self.amplitude))(x)
                elif neuron_type == "adaptivelifrate":
                    x = nengo_dl.Layer(nengo.AdaptiveLIFRate(amplitude=self.amplitude))(x)
                elif neuron_type == "izhikevich":
                    x = nengo_dl.Layer(nengo.Izhikevich(amplitude=self.amplitude))(x)
                elif neuron_type == "softlifrate":
                    x = nengo_dl.Layer(nengo_dl.neurons.SoftLIFRate(amplitude=self.amplitude))(x)
                elif neuron_type == None:  # default neuron_type = LIF
                    x = nengo_dl.Layer(nengo.LIF(amplitude=self.amplitude))(x)
        output_shape = [dense_num, 1]
        print('convert Dense finish')
        return model, output_shape, x  # x를 return 하면서 모델을 계속 쌓아감

    # neuron이 op_type이 어떤건지 - node들 지나다니면서 뒤짐
    def get_neuronType(self, node_index, onnx_model_graph_node):
        node_len = len(onnx_model_graph_node)
        for index in range(node_index, node_len):
            node_info = onnx_model_graph_node[index]
            op_type = node_info.op_type.lower()
            if op_type == "lif" or op_type == "lifrate" or op_type == "adaptivelif" or op_type == "adaptivelifrate" or op_type == "izhikevich" or op_type == "softmax" or op_type == "softlifrate":
                return op_type
        return None

    # 모델 반환
    def get_model(self):
        return self.model

    # input
    def get_inputProbe(self):
        return self.inp

    def get_endLayer(self):
        return self.end_layer

    def get_filterNum(self, node_info, onnx_model_graph):
        weight_name = node_info.input[1]
        for m in range(len(onnx_model_graph.initializer)):
            name = onnx_model_graph.initializer[m].name
            for n in range(len(node_info.input)):
                if node_info.input[n] == name and name == weight_name:
                    shape = onnx_model_graph.initializer[m].dims
                    return shape[0]
        return None

    def get_dense_num(self, node_info, onnx_model_graph):
        weight_name = node_info.input[1]
        for m in range(len(onnx_model_graph.initializer)):
            name = onnx_model_graph.initializer[m].name
            for n in range(len(node_info.input)):
                if node_info.input[n] == name and name == weight_name:
                    shape = onnx_model_graph.initializer[m].dims
                    return shape[1]
        return None


# 일반
def objective(outputs, targets):
	return tf.keras.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets)

def classification_error(outputs, targets):
	return 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))

def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])