# nengo

import os

import nengo
import nengo_dl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import onnx
import onnx.numpy_helper as numpy_helper
import warnings
warnings.filterwarnings('ignore')

try:
    import requests

    has_requests = True
except ImportError:
    has_requests = False

import nengo_loihi


def save_npz(onnx_model, npz_name):
    graph = onnx_model.graph
    initializers = []

    neurons = 0

    for init in graph.initializer:
        initializers.append([init.name, numpy_helper.to_array(init)])

    # conv, dense 가중치 순서가 뒤집혀있기 때문에 올바르게 정렬
    initializers = sorted(initializers, key=lambda x: (-len(x[1].shape)))

    temp = []

    for (name, weight) in initializers:
        if len(weight.shape) == 4:  # conv
            weight = np.transpose(weight, (2, 3, 1, 0))

        if len(weight.shape) == 2:  # dense
            weight = np.transpose(weight, (1, 0))

        temp.append(weight)
        neurons += weight.shape[0]

    if graph.node[-1].op_type != "relu" and graph.node[-1].op_type != "softmax":
        neurons -= weight.shape[0]  # 마지막 층이 활성화 함수 없이 dense로 끝나면 뉴런은 빼야함

    last = np.random.normal(1, 1, neurons)
    temp.append(last)

    np.savez_compressed(npz_name, *temp)


def download(fname, drive_id):
    """Download a file from Google Drive.

    Adapted from https://stackoverflow.com/a/39225039/1306923
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    if os.path.exists(fname):
        return
    if not has_requests:
        link = "https://drive.google.com/open?id=%s" % drive_id
        raise RuntimeError(
            "Cannot find '%s'. Download the file from\n  %s\n"
            "and place it in %s." % (fname, link, os.getcwd())
        )

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={"id": drive_id}, stream=True)
    token = get_confirm_token(response)
    if token is not None:
        params = {"id": drive_id, "confirm": token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, fname)


# load mnist dataset
(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

# flatten images
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# plot some examples
for i in range(3):
    plt.figure()
    plt.imshow(np.reshape(train_images[i], (28, 28)))
    plt.axis("off")
    plt.title(str(train_labels[i]))

dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 200  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape

with nengo.Network(seed=0) as net:
    nengo_loihi.add_params(net)
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=0.005)

    onnx_name = "softlif_dense"
    onnx_model = onnx.load("result/" + onnx_name + ".onnx")
    print("load {}.onnx".format(onnx_name))
    save_npz(onnx_model, "result/" + onnx_name + ".npz")
    graph = onnx_model.graph

    input_size = numpy_helper.to_array(graph.initializer[0]).shape[0]
    inp = nengo.Node(nengo.processes.PresentInput(test_images, presentation_time), size_out=input_size)
    pre_layer = inp

    for i in range(len(graph.initializer)):
        n_nodes = numpy_helper.to_array(graph.initializer[i]).shape[1]

        if i == 0:  #문제없음
            layer = nengo.Ensemble(n_neurons=n_nodes, dimensions=1, neuron_type=neuron_type)
            net.config[layer].on_chip = False
            nengo.Connection(pre_layer, layer.neurons, transform=nengo_dl.dists.Glorot())
            pre_layer = layer

        elif i != 0 and i != len(graph.initializer)-1:
            layer = nengo.Ensemble(n_neurons=n_nodes, dimensions=1, neuron_type=neuron_type)
            nengo.Connection(pre_layer.neurons, layer.neurons, transform=nengo_dl.dists.Glorot())
            pre_layer = layer

        else:
            out = nengo.Node(size_in=n_nodes)
            nengo.Connection(pre_layer.neurons, out, transform=nengo_dl.dists.Glorot())

            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01), label="out_p_filt")

# set up training data, adding the time dimension (with size 1)
minibatch_size = 200
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
n_steps = int(presentation_time / dt)
test_images = np.tile(test_images[:minibatch_size*2, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[:minibatch_size*2, None, None], (1, n_steps, 1))


def classification_accuracy(y_true, y_pred):
    return 100 * tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


do_training = False

with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print(
            "accuracy before training: %.2f%%"
            % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
        )

        # run training
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
        )
        sim.fit(train_images, train_labels, epochs=10)

        sim.compile(loss={out_p_filt: classification_accuracy})
        print(
            "accuracy after training: %.2f%%"
            % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
        )

        sim.save_params("./mnist_params")
    else:
        download("mnist_params.npz", "1geZoS-Nz-u_XeeDv3cdZgNjUxDOpgXe5")
        #sim.load_params("./mnist_params")

        sim.load_params("result/softlif_dense")

        sim.compile(loss={out_p_filt: classification_accuracy})
        # print(
        #     "nengo_dl accuracy load after training: %.2f%%"
        #     % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
        # )

    # store trained parameters back into the network
    sim.freeze_params(net)

for conn in net.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print(
            "accuracy w/ synapse: %.2f%%"
            % sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"]
        )

n_presentations = 100  # 네트워크에 입력하는 테스트 이미지 갯수?

# if running on Loihi, increase the max input spikes per step
hw_opts = dict(snip_max_spikes_per_step=120)
with nengo_loihi.Simulator(
    net,
    dt=dt,
    precompute=False,
    hardware_options=hw_opts,
) as sim:
    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification accuracy
    step = int(presentation_time / dt)

    output = sim.data[out_p_filt][step - 1 :: step]
    correct = 100 * np.mean(
        np.argmax(output, axis=-1) == test_labels[:n_presentations, 0, 0]
    )
    print("loihi accuracy: %.2f%%" % correct)

n_plots = 10
plt.figure()

plt.subplot(2, 1, 1)
images = test_images.reshape(-1, 28, 28, 1)[::step]
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj * n_plots, nc), dtype=images.dtype)
for i, image in enumerate(images[:n_plots]):
    allimage[:, i * nj : (i + 1) * nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
plt.imshow(allimage, aspect="auto", interpolation="none", cmap="gray")

plt.subplot(2, 1, 2)
plt.plot(sim.trange()[: n_plots * step], sim.data[out_p_filt][: n_plots * step])
plt.legend(["%d" % i for i in range(10)], loc="best")
# plt.show()