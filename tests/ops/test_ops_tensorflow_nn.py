import os
import unittest

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "tensorflow"


from tensorlayerx.backend.ops.tensorflow_nn import *
from tests.utils import CustomTestCase


class TestTensorFlowNN(CustomTestCase):

    def test_padding_format(self):
        self.assertEqual(padding_format("same"), "SAME")
        self.assertEqual(padding_format("SAME"), "SAME")
        self.assertEqual(padding_format("valid"), "VALID")
        self.assertEqual(padding_format("VALID"), "VALID")
        self.assertEqual(padding_format(2), 2)
        self.assertEqual(padding_format((1, 2)), (1, 2))
        with self.assertRaises(Exception):
            padding_format("unsupported")

    def test_channel_format(self):
        self.assertEqual(channel_format("channels_last", "1d"), "NWC")
        self.assertEqual(channel_format("channels_first", "1d"), "NCW")
        self.assertEqual(channel_format("channels_last", "2d"), "NHWC")
        self.assertEqual(channel_format("channels_first", "2d"), "NCHW")
        self.assertEqual(channel_format("channels_last", "3d"), "NDHWC")
        self.assertEqual(channel_format("channels_first", "3d"), "NCDHW")
        with self.assertRaises(Exception):
            channel_format("unsupported", "2d")

    def test_preprocess_padding(self):
        self.assertEqual(preprocess_padding(2, "1d", "NWC"), [[0, 0], [2, 2], [0, 0]])
        self.assertEqual(preprocess_padding((1, 2), "2d", "NHWC"), [[0, 0], [1, 1], [2, 2], [0, 0]])
        self.assertEqual(preprocess_padding(2, "3d", "NDHWC"), [[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]])
        with self.assertRaises(RuntimeError):
            preprocess_padding((1, 2, 3, 4), "2d", "NHWC")

    def test_nchw_to_nhwc(self):
        x = tf.random.normal([1, 3, 32, 32])
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 32, 32, 3))

    def test_nhwc_to_nchw(self):
        x = tf.random.normal([1, 32, 32, 3])
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 3, 32, 32))

    def test_relu(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        y = relu(x)
        self.assertTrue(np.array_equal(y.numpy(), [0.0, 0.0, 1.0]))

    def test_elu(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        y = elu(x)
        self.assertTrue(np.allclose(y.numpy(), [-0.63212055, 0.0, 1.0]))

    def test_relu6(self):
        x = tf.constant([-1.0, 0.0, 6.0, 7.0])
        y = relu6(x)
        self.assertTrue(np.array_equal(y.numpy(), [0.0, 0.0, 6.0, 6.0]))

    def test_leaky_relu(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        y = leaky_relu(x)
        self.assertTrue(np.allclose(y.numpy(), [-0.2, 0.0, 1.0]))

    def test_softplus(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        softplus = Softplus()
        y = softplus(x)
        self.assertTrue(np.allclose(y.numpy(), [0.31326166, 0.69314718, 1.31326169]))

    def test_tanh(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        tanh = Tanh()
        y = tanh(x)
        self.assertTrue(np.allclose(y.numpy(), [-0.76159416, 0.0, 0.76159416]))

    def test_sigmoid(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        y = sigmoid(x)
        self.assertTrue(np.allclose(y.numpy(), [0.26894142, 0.5, 0.73105858]))

    def test_softmax(self):
        x = tf.constant([1.0, 2.0, 3.0])
        y = softmax(x)
        self.assertTrue(np.allclose(y.numpy(), [0.09003057, 0.24472847, 0.66524096]))

    def test_gelu(self):
        x = tf.constant([-1.0, 0.0, 1.0])
        y = gelu(x)
        self.assertTrue(np.allclose(y.numpy(), [-0.15865529, 0.0, 0.84134471]))

    def test_dropout(self):
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        dropout = Dropout(p=0.5, seed=42)
        y = dropout(x)
        self.assertEqual(y.shape, x.shape)
        # Check that some elements are zeroed out
        self.assertTrue(np.any(y.numpy() == 0.0))
        # Check that the non-zero elements are scaled by 1/(1-p)
        scale_factor = 1 / (1 - 0.5)
        non_zero_elements = y.numpy()[y.numpy() != 0.0]
        self.assertTrue(np.allclose(non_zero_elements, x.numpy()[y.numpy() != 0.0] * scale_factor))

    def test_bias_add(self):
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        bias = tf.constant([1.0, 2.0])
        y = bias_add(x, bias)
        self.assertTrue(np.array_equal(y.numpy(), [[2.0, 4.0], [4.0, 6.0]]))

    def test_conv1d(self):
        x = tf.random.normal([1, 10, 3])
        filters = tf.random.normal([3, 3, 2])
        y = conv1d(x, filters, stride=1, padding="SAME")
        self.assertEqual(y.shape, (1, 10, 2))

    def test_conv2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        filters = tf.random.normal([3, 3, 3, 16])
        y = conv2d(x, filters, strides=1, padding="SAME")
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_conv3d(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        filters = tf.random.normal([3, 3, 3, 3, 16])
        y = conv3d(x, filters, strides=1, padding="SAME")
        self.assertEqual(y.shape, (1, 16, 16, 16, 16))

    def test_lrn(self):
        x = tf.random.normal([1, 32, 32, 3])
        y = lrn(x, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5)
        self.assertEqual(y.shape, x.shape)

    def test_moments(self):
        x = tf.random.normal([1, 32, 32, 3])
        mean, variance = moments(x, axes=[0, 1, 2])
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(variance.shape, (3,))

    def test_max_pool1d(self):
        x = tf.random.normal([1, 10, 3])
        max_pool_1d = MaxPool1d(ksize=2, strides=2, padding="SAME", return_mask=False)
        y = max_pool_1d(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_max_pool(self):
        x = tf.random.normal([1, 32, 32, 3])
        max_pool_2d = MaxPool(ksize=2, strides=2, padding="SAME", return_mask=False)
        y = max_pool_2d(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_max_pool1d_function(self):
        x = tf.random.normal([1, 10, 3])
        y = max_pool1d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 5, 3))

    def test_max_pool2d_function(self):
        x = tf.random.normal([1, 32, 32, 3])
        y = max_pool2d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_max_pool3d_function(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        y = max_pool3d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_avg_pool1d(self):
        x = tf.random.normal([1, 10, 3])
        avg_pool_1d = AvgPool1d(ksize=2, strides=2, padding="SAME")
        y = avg_pool_1d(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_avg_pool(self):
        x = tf.random.normal([1, 32, 32, 3])
        avg_pool_2d = AvgPool(ksize=2, strides=2, padding="SAME")
        y = avg_pool_2d(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_avg_pool1d_function(self):
        x = tf.random.normal([1, 10, 3])
        y = avg_pool1d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 5, 3))

    def test_avg_pool2d_function(self):
        x = tf.random.normal([1, 32, 32, 3])
        y = avg_pool2d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_avg_pool3d_function(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        y = avg_pool3d(x, kernel_size=2, stride=2, padding="SAME")
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_pool(self):
        x = tf.random.normal([1, 32, 32, 3])
        y = pool(x, window_shape=[2, 2], pooling_type="MAX", strides=[2, 2], padding="SAME")
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_depthwise_conv2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        filters = tf.random.normal([3, 3, 3, 1])
        y = depthwise_conv2d(x, filters, strides=[1, 1, 1, 1], padding="SAME")
        self.assertEqual(y.shape, (1, 32, 32, 3))

    def test_conv1d_transpose(self):
        x = tf.random.normal([1, 10, 3])
        filters = tf.random.normal([3, 2, 3])
        y = conv1d_transpose(x, filters, output_shape=[1, 10, 2], strides=1, padding="SAME")
        self.assertEqual(y.shape, (1, 10, 2))

    def test_conv2d_transpose(self):
        x = tf.random.normal([1, 32, 32, 3])
        filters = tf.random.normal([3, 3, 16, 3])
        y = conv2d_transpose(x, filters, output_shape=[1, 32, 32, 16], strides=1, padding="SAME")
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_conv3d_transpose(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        filters = tf.random.normal([3, 3, 3, 16, 3])
        y = conv3d_transpose(x, filters, output_shape=[1, 16, 16, 16, 16], strides=1, padding="SAME")
        self.assertEqual(y.shape, (1, 16, 16, 16, 16))

    def test_batch_normalization(self):
        x = tf.random.normal([1, 32, 32, 3])
        mean = tf.constant([0.5, 0.5, 0.5])
        variance = tf.constant([0.25, 0.25, 0.25])
        offset = tf.constant([0.0, 0.0, 0.0])
        scale = tf.constant([1.0, 1.0, 1.0])
        y = batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-5, data_format="channels_last")
        self.assertEqual(y.shape, (1, 32, 32, 3))

    def test_group_conv2d(self):
        x = tf.random.normal([1, 32, 32, 16])
        filters = tf.random.normal([3, 3, 4, 16])
        group_conv2d = GroupConv2D(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=16, k_size=3, groups=4)
        y = group_conv2d(x, filters)
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_separable_conv1d(self):
        x = tf.random.normal([1, 10, 3])
        depthwise_filters = tf.random.normal([3, 1, 3])
        pointwise_filters = tf.random.normal([1, 3, 2])
        separable_conv1d = SeparableConv1D(stride=1, padding="SAME", data_format="NWC", dilations=1, out_channel=2, k_size=3, in_channel=3, depth_multiplier=1)
        y = separable_conv1d(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 10, 2))

    def test_separable_conv2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        depthwise_filters = tf.random.normal([3, 3, 3, 1])
        pointwise_filters = tf.random.normal([1, 1, 3, 16])
        separable_conv2d = SeparableConv2D(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=16, k_size=3, in_channel=3, depth_multiplier=1)
        y = separable_conv2d(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_adaptive_mean_pool1d(self):
        x = tf.random.normal([1, 10, 3])
        adaptive_mean_pool1d = AdaptiveMeanPool1D(output_size=5, data_format="NWC")
        y = adaptive_mean_pool1d(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_mean_pool2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        adaptive_mean_pool2d = AdaptiveMeanPool2D(output_size=(16, 16), data_format="NHWC")
        y = adaptive_mean_pool2d(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_adaptive_mean_pool3d(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        adaptive_mean_pool3d = AdaptiveMeanPool3D(output_size=(8, 8, 8), data_format="NDHWC")
        y = adaptive_mean_pool3d(x)
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_adaptive_max_pool1d(self):
        x = tf.random.normal([1, 10, 3])
        adaptive_max_pool1d = AdaptiveMaxPool1D(output_size=5, data_format="NWC")
        y = adaptive_max_pool1d(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_max_pool2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        adaptive_max_pool2d = AdaptiveMaxPool2D(output_size=(16, 16), data_format="NHWC")
        y = adaptive_max_pool2d(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_adaptive_max_pool3d(self):
        x = tf.random.normal([1, 16, 16, 16, 3])
        adaptive_max_pool3d = AdaptiveMaxPool3D(output_size=(8, 8, 8), data_format="NDHWC")
        y = adaptive_max_pool3d(x)
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_binary_conv2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        filters = tf.random.normal([3, 3, 3, 16])
        binary_conv2d = BinaryConv2D(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=16, k_size=3, in_channel=3)
        y = binary_conv2d(x, filters)
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_dorefa_conv2d(self):
        x = tf.random.normal([1, 32, 32, 3])
        filters = tf.random.normal([3, 3, 3, 16])
        dorefa_conv2d = DorefaConv2D(bitW=1, bitA=1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=16, k_size=3, in_channel=3)
        y = dorefa_conv2d(x, filters)
        self.assertEqual(y.shape, (1, 32, 32, 16))

    def test_rnncell(self):
        input = tf.random.normal([1, 3])
        h = tf.random.normal([1, 3])
        weight_ih = tf.random.normal([3, 3])
        weight_hh = tf.random.normal([3, 3])
        bias_ih = tf.random.normal([3])
        bias_hh = tf.random.normal([3])
        cell = rnncell(weight_ih, weight_hh, bias_ih, bias_hh, act="relu")
        h, _ = cell(input, h)
        self.assertEqual(h.shape, (1, 3))

    def test_lstmcell(self):
        input = tf.random.normal([1, 3])
        h = tf.random.normal([1, 3])
        c = tf.random.normal([1, 3])
        weight_ih = tf.random.normal([3, 3])
        weight_hh = tf.random.normal([3, 3])
        bias_ih = tf.random.normal([3])
        bias_hh = tf.random.normal([3])
        cell = lstmcell(weight_ih, weight_hh, bias_ih, bias_hh)
        h, _, c = cell(input, h, c)
        self.assertEqual(h.shape, (1, 3))
        self.assertEqual(c.shape, (1, 3))

    def test_grucell(self):
        input = tf.random.normal([1, 3])
        h = tf.random.normal([1, 3])
        weight_ih = tf.random.normal([3, 3])
        weight_hh = tf.random.normal([3, 3])
        bias_ih = tf.random.normal([3])
        bias_hh = tf.random.normal([3])
        cell = grucell(weight_ih, weight_hh, bias_ih, bias_hh)
        h, _ = cell(input, h)
        self.assertEqual(h.shape, (1, 3))

    def test_rnnbase_lstm(self):
        input = tf.random.normal([5, 1, 3])
        h = tf.random.normal([1, 1, 3])
        c = tf.random.normal([1, 1, 3])
        w_ih = [tf.random.normal([3, 3])]
        w_hh = [tf.random.normal([3, 3])]
        b_ih = [tf.random.normal([3])]
        b_hh = [tf.random.normal([3])]
        rnn = rnnbase("LSTM", 3, 3, 1, True, False, 0.0, False, True, w_ih, w_hh, b_ih, b_hh)
        y, (new_h, new_c) = rnn(input, (h, c))
        self.assertEqual(y.shape, (5, 1, 3))
        self.assertEqual(new_h.shape, (1, 1, 3))
        self.assertEqual(new_c.shape, (1, 1, 3))

    def test_rnnbase_gru(self):
        input = tf.random.normal([5, 1, 3])
        h = tf.random.normal([1, 1, 3])
        w_ih = [tf.random.normal([3, 3])]
        w_hh = [tf.random.normal([3, 3])]
        b_ih = [tf.random.normal([3])]
        b_hh = [tf.random.normal([3])]
        rnn = rnnbase("GRU", 3, 3, 1, True, False, 0.0, False, True, w_ih, w_hh, b_ih, b_hh)
        y, new_h = rnn(input, h)
        self.assertEqual(y.shape, (5, 1, 3))
        self.assertEqual(new_h.shape, (1, 1, 3))

    def test_rnnbase_rnn_tanh(self):
        input = tf.random.normal([5, 1, 3])
        h = tf.random.normal([1, 1, 3])
        w_ih = [tf.random.normal([3, 3])]
        w_hh = [tf.random.normal([3, 3])]
        b_ih = [tf.random.normal([3])]
        b_hh = [tf.random.normal([3])]
        rnn = rnnbase("RNN_TANH", 3, 3, 1, True, False, 0.0, False, True, w_ih, w_hh, b_ih, b_hh)
        y, new_h = rnn(input, h)
        self.assertEqual(y.shape, (5, 1, 3))
        self.assertEqual(new_h.shape, (1, 1, 3))

    def test_rnnbase_rnn_relu(self):
        input = tf.random.normal([5, 1, 3])
        h = tf.random.normal([1, 1, 3])
        w_ih = [tf.random.normal([3, 3])]
        w_hh = [tf.random.normal([3, 3])]
        b_ih = [tf.random.normal([3])]
        b_hh = [tf.random.normal([3])]
        rnn = rnnbase("RNN_RELU", 3, 3, 1, True, False, 0.0, False, True, w_ih, w_hh, b_ih, b_hh)
        y, new_h = rnn(input, h)
        self.assertEqual(y.shape, (5, 1, 3))
        self.assertEqual(new_h.shape, (1, 1, 3))

    def test_layernorm(self):
        input = tf.random.normal([2, 3, 4])
        gamma = tf.ones([4])
        beta = tf.zeros([4])
        layer_norm = layernorm([4], gamma, beta, 1e-5, [2, 3, 4])
        output = layer_norm(input)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_multiheadattention(self):
        q = tf.random.normal([5, 1, 8])
        k = tf.random.normal([5, 1, 8])
        v = tf.random.normal([5, 1, 8])
        q_weight = tf.random.normal([8, 8])
        k_weight = tf.random.normal([8, 8])
        v_weight = tf.random.normal([8, 8])
        out_weight = tf.random.normal([8, 8])
        q_bias = tf.random.normal([8])
        k_bias = tf.random.normal([8])
        v_bias = tf.random.normal([8])
        out_bias = tf.random.normal([8])
        mha = multiheadattention(embed_dim=8, num_heads=2, dropout=0.1, batch_first=False, need_weights=False, q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, out_weight=out_weight, q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias, train=True)
        output, _ = mha(q, k, v, None, None)
        self.assertEqual(output.shape, (5, 1, 8))

    def test_binary_dense(self):
        inputs = tf.random.normal([2, 3])
        weights = tf.random.normal([3, 4])
        bias = tf.random.normal([4])
        binary_dense = BinaryDense(weights, bias)
        outputs = binary_dense(inputs)
        self.assertEqual(outputs.shape, (2, 4))

    def test_dorefa_dense(self):
        inputs = tf.random.normal([2, 3])
        weights = tf.random.normal([3, 4])
        bias = tf.random.normal([4])
        dorefa_dense = DorefaDense(weights, bias, bitW=1, bitA=1)
        outputs = dorefa_dense(inputs)
        self.assertEqual(outputs.shape, (2, 4))

    def test_ternary_dense(self):
        inputs = tf.random.normal([2, 3])
        weights = tf.random.normal([3, 4])
        bias = tf.random.normal([4])
        ternary_dense = TernaryDense(weights, bias)
        outputs = ternary_dense(inputs)
        self.assertEqual(outputs.shape, (2, 4))

    def test_quan_dense(self):
        inputs = tf.random.normal([2, 3])
        weights = tf.random.normal([3, 4])
        bias = tf.random.normal([4])
        quan_dense = QuanDense(weights, bias, bitW=2, bitA=2)
        outputs = quan_dense(inputs)
        self.assertEqual(outputs.shape, (2, 4))

    def test_quan_dense_bn(self):
        inputs = tf.random.normal([2, 3])
        weights = tf.random.normal([3, 4])
        scale_para = tf.random.normal([4])
        offset_para = tf.random.normal([4])
        moving_mean = tf.random.normal([4])
        moving_variance = tf.random.normal([4])
        quan_dense_bn = QuanDenseBn(weights, scale_para, offset_para, moving_mean, moving_variance, decay=0.9, bitW=2, bitA=2, epsilon=1e-5, is_train=True)
        outputs = quan_dense_bn(inputs)
        self.assertEqual(outputs.shape, (2, 4))

    def test_ternary_conv(self):
        inputs = tf.random.normal([1, 32, 32, 3])
        weights = tf.random.normal([3, 3, 3, 16])
        ternary_conv = TernaryConv(weights, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1])
        outputs = ternary_conv(inputs)
        self.assertEqual(outputs.shape, (1, 32, 32, 16))

    def test_quan_conv(self):
        inputs = tf.random.normal([1, 32, 32, 3])
        weights = tf.random.normal([3, 3, 3, 16])
        quan_conv = QuanConv(weights, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], bitW=2, bitA=2)
        outputs = quan_conv(inputs)
        self.assertEqual(outputs.shape, (1, 32, 32, 16))

    def test_quan_conv_bn(self):
        inputs = tf.random.normal([1, 32, 32, 3])
        weights = tf.random.normal([3, 3, 3, 16])
        scale_para = tf.random.normal([16])
        offset_para = tf.random.normal([16])
        moving_mean = tf.random.normal([16])
        moving_variance = tf.random.normal([16])
        quan_conv_bn = QuanConvBn(weights, scale_para, offset_para, moving_mean, moving_variance, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], bitW=2, bitA=2, decay=0.9, epsilon=1e-5, is_train=True)
        outputs = quan_conv_bn(inputs)
        self.assertEqual(outputs.shape, (1, 32, 32, 16))

    def test_prelu(self):
        inputs = tf.random.normal([1, 32, 32, 3])
        weight = tf.random.normal([3])
        prelu_layer = PReLU(data_format="channels_last")
        outputs = prelu_layer(inputs, weight)
        self.assertEqual(outputs.shape, (1, 32, 32, 3))

    def test_hardsigmoid(self):
        inputs = tf.constant([-1.0, 0.0, 1.0])
        outputs = hardsigmoid(inputs)
        self.assertTrue(np.allclose(outputs.numpy(), [0.0, 0.5, 1.0]))

    def test_hardswish(self):
        inputs = tf.constant([-1.0, 0.0, 1.0])
        outputs = hardswish(inputs)
        self.assertTrue(np.allclose(outputs.numpy(), [0.0, 0.0, 0.33333334]))

    def test_swish(self):
        inputs = tf.constant([-1.0, 0.0, 1.0])
        outputs = swish(inputs)
        self.assertTrue(np.allclose(outputs.numpy(), [-0.26894142, 0.0, 0.73105858]))

    def test_linear(self):
        inputs = tf.random.normal([2, 3])
        weight = tf.random.normal([4, 3])
        bias = tf.random.normal([4])
        outputs = linear(inputs, weight, bias)
        self.assertEqual(outputs.shape, (2, 4))

    def test_unfold(self):
        inputs = tf.random.normal([1, 3, 32, 32])
        outputs = unfold(inputs, kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        self.assertEqual(outputs.shape, (1, 27, 1024))


if __name__ == "__main__":
    unittest.main()
