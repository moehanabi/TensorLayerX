import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "mindspore"
import numpy as np
from mindspore import Tensor

from tensorlayerx.backend.ops.mindspore_nn import *
from tests.utils import CustomTestCase

set_device(device="Ascend", id=0)
print(get_device())


class TestMindsporeNN(CustomTestCase):

    def test_padding_format(self):
        self.assertEqual(padding_format("SAME"), "same")
        self.assertEqual(padding_format("same"), "same")
        self.assertEqual(padding_format("VALID"), "valid")
        self.assertEqual(padding_format("valid"), "valid")
        self.assertEqual(padding_format(2), 2)
        self.assertEqual(padding_format((1, 2)), (1, 2))
        with self.assertRaises(Exception):
            padding_format("unsupported")

    def test_preprocess_padding(self):
        self.assertEqual(preprocess_padding(2, "1d"), (0, 0, 2, 2))
        self.assertEqual(preprocess_padding((1, 2), "2d"), (1, 1, 2, 2))
        self.assertEqual(preprocess_padding((1, 2, 3), "3d"), (1, 1, 2, 2, 3, 3))
        with self.assertRaises(RuntimeError):
            preprocess_padding(2, "4d")

    def test_check_padding(self):
        with self.assertRaises(RuntimeError):
            check_padding((1, 2), "1d")
        with self.assertRaises(RuntimeError):
            check_padding((1, 2, 3), "2d")
        with self.assertRaises(RuntimeError):
            check_padding((1, 2, 3, 4), "3d")

    def test_preprocess_1d_format(self):
        self.assertEqual(preprocess_1d_format("channels_last", "SAME"), ("NWC", "same"))
        self.assertEqual(preprocess_1d_format("NCW", "VALID"), ("NCW", "valid"))
        with self.assertRaises(Exception):
            preprocess_1d_format("unsupported", "SAME")

    def test_preprocess_2d_format(self):
        self.assertEqual(preprocess_2d_format("channels_last", "SAME"), ("NHWC", "same"))
        self.assertEqual(preprocess_2d_format("NCHW", "VALID"), ("NCHW", "valid"))
        with self.assertRaises(Exception):
            preprocess_2d_format("unsupported", "SAME")

    def test_preprocess_3d_format(self):
        self.assertEqual(preprocess_3d_format("channels_last", "SAME"), ("NDHWC", "same"))
        self.assertEqual(preprocess_3d_format("NCDHW", "VALID"), ("NCDHW", "valid"))
        with self.assertRaises(Exception):
            preprocess_3d_format("unsupported", "SAME")

    def test_nchw_to_nhwc(self):
        x = Tensor(np.ones((1, 2, 3, 4)))
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 3, 4, 2))

    def test_nhwc_to_nchw(self):
        x = Tensor(np.ones((1, 3, 4, 2)))
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 2, 3, 4))

    def test_relu(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = relu(x)
        np.testing.assert_array_equal(y.asnumpy(), np.array([0, 0, 1], dtype=np.float32))

    def test_elu(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = elu(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([-0.6321, 0, 1], dtype=np.float32), decimal=4)

    def test_relu6(self):
        x = Tensor(np.array([-1, 0, 1, 6, 7], dtype=np.float32))
        y = relu6(x)
        np.testing.assert_array_equal(y.asnumpy(), np.array([0, 0, 1, 6, 6], dtype=np.float32))

    def test_leaky_relu(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = leaky_relu(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([-0.2, 0, 1], dtype=np.float32), decimal=4)

    def test_sigmoid(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = sigmoid(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([0.2689, 0.5, 0.7311], dtype=np.float32), decimal=4)

    def test_softmax(self):
        x = Tensor(np.array([1, 2, 3], dtype=np.float32))
        y = softmax(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([0.0900, 0.2447, 0.6652], dtype=np.float32), decimal=4)

    def test_gelu(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = gelu(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([-0.1588, 0, 0.8413], dtype=np.float32), decimal=4)

    def test_dropout(self):
        x = Tensor(np.ones((2, 2), dtype=np.float32))
        dropout = Dropout(p=0.5, seed=0)
        y = dropout.construct(x)
        self.assertEqual(y.shape, x.shape)

    def test_bias_add(self):
        x = np.ones((1, 2, 3, 4), dtype=np.float32)
        bias = np.ones((4,), dtype=np.float32)
        bias_add = BiasAdd(data_format="channels_last")
        y = bias_add.construct(x, bias)
        self.assertEqual(y.shape, x.shape)

    def test_conv1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3), dtype=np.float32)
        conv1d = Conv1D(stride=1, padding="VALID", data_format="NWC", dilations=1, out_channel=3, k_size=3)
        y = conv1d.construct(x, filters)
        self.assertEqual(y.shape, (1, 8, 3))

    def test_conv2d(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3), dtype=np.float32)
        conv2d = Conv2D(strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3))
        y = conv2d.construct(x, filters)
        self.assertEqual(y.shape, (1, 1, 1, 3))

    def test_conv3d(self):
        x = np.ones((1, 3, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3, 3), dtype=np.float32)
        conv3d = Conv3D(strides=(1, 1, 1, 1, 1), padding="VALID", data_format="NCDHW", dilations=(1, 1, 1, 1, 1), out_channel=3, k_size=(3, 3, 3))
        y = conv3d.construct(x, filters)
        self.assertEqual(y.shape, (1, 1, 1, 1, 3))

    def test_max_pool1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        max_pool1d = MaxPool1d(ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NWC")
        y = max_pool1d.construct(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_max_pool(self):
        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        max_pool = MaxPool(ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NHWC")
        y = max_pool.construct(x)
        self.assertEqual(y.shape, (1, 2, 2, 3))

    def test_avg_pool1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        avg_pool1d = AvgPool1d(ksize=2, strides=2, padding="VALID", data_format="NWC")
        y = avg_pool1d.construct(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_avg_pool(self):
        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        avg_pool = AvgPool(ksize=2, strides=2, padding="VALID", data_format="NHWC")
        y = avg_pool.construct(x)
        self.assertEqual(y.shape, (1, 2, 2, 3))

    def test_depthwise_conv2d(self):
        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 1), dtype=np.float32)
        point_filters = np.ones((1, 1, 3, 3), dtype=np.float32)
        depthwise_conv2d = DepthwiseConv2d(strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1, 1, 1), ksize=(3, 3), channel_multiplier=1, in_channels=3)
        y = depthwise_conv2d.construct(x, filters, point_filters)
        self.assertEqual(y.shape, (1, 2, 2, 3))

    def test_conv1d_transpose(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3), dtype=np.float32)
        conv1d_transpose = Conv1d_transpose(stride=1, padding="VALID", data_format="NCW", dilations=1, out_channel=3, k_size=3, in_channels=3)
        y = conv1d_transpose.construct(x, filters)
        self.assertEqual(y.shape, (1, 12, 3))

    def test_conv2d_transpose(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3), dtype=np.float32)
        conv2d_transpose = Conv2d_transpose(strides=(1, 1, 1, 1), padding="VALID", data_format="NCHW", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3), in_channels=3)
        y = conv2d_transpose.construct(x, filters, output_size=None)
        self.assertEqual(y.shape, (1, 3, 5, 5))

    def test_conv3d_transpose(self):
        x = np.ones((1, 3, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3, 3), dtype=np.float32)
        conv3d_transpose = Conv3d_transpose(strides=(1, 1, 1, 1, 1), padding="VALID", data_format="NCDHW", dilations=(1, 1, 1, 1, 1), out_channel=3, k_size=(3, 3, 3), in_channels=3)
        y = conv3d_transpose.construct(x, filters)
        self.assertEqual(y.shape, (1, 3, 5, 5, 5))

    def test_batch_norm(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        gamma = np.ones((3,), dtype=np.float32)
        beta = np.zeros((3,), dtype=np.float32)
        moving_mean = np.zeros((3,), dtype=np.float32)
        moving_variance = np.ones((3,), dtype=np.float32)
        batch_norm = BatchNorm(num_features=3, epsilon=1e-5, decay=0.9, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_variance, is_train=True, data_format="NCHW")
        y = batch_norm.construct(x)
        self.assertEqual(y.shape, x.shape)

    def test_group_conv2d(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 1, 3), dtype=np.float32)
        group_conv2d = GroupConv2D(strides=(1, 1, 1, 1), padding="VALID", data_format="NCHW", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3), groups=3)
        y = group_conv2d.construct(x, filters)
        self.assertEqual(y.shape, (1, 3, 1, 1))

    def test_separable_conv1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        depthwise_filters = np.ones((3, 3, 1), dtype=np.float32)
        pointwise_filters = np.ones((1, 1, 3), dtype=np.float32)
        separable_conv1d = SeparableConv1D(stride=1, padding="VALID", data_format="NWC", dilations=1, out_channel=3, k_size=3, in_channel=3, depth_multiplier=1)
        y = separable_conv1d.construct(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 8, 3))

    def test_separable_conv2d(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        depthwise_filters = np.ones((3, 3, 3, 1), dtype=np.float32)
        pointwise_filters = np.ones((1, 1, 3, 3), dtype=np.float32)
        separable_conv2d = SeparableConv2D(strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3), in_channel=3, depth_multiplier=1)
        y = separable_conv2d.construct(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 1, 1, 3))

    def test_adaptive_mean_pool1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        adaptive_mean_pool1d = AdaptiveMeanPool1D(output_size=5, data_format="NWC")
        y = adaptive_mean_pool1d.construct(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_mean_pool2d(self):
        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        adaptive_mean_pool2d = AdaptiveMeanPool2D(output_size=(2, 2), data_format="NHWC")
        y = adaptive_mean_pool2d.construct(x)
        self.assertEqual(y.shape, (1, 2, 2, 3))

    def test_adaptive_max_pool1d(self):
        x = np.ones((1, 10, 3), dtype=np.float32)
        adaptive_max_pool1d = AdaptiveMaxPool1D(output_size=5, data_format="NWC")
        y = adaptive_max_pool1d.construct(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_max_pool2d(self):
        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        adaptive_max_pool2d = AdaptiveMaxPool2D(output_size=(2, 2), data_format="NHWC")
        y = adaptive_max_pool2d.construct(x)
        self.assertEqual(y.shape, (1, 2, 2, 3))

    def test_binary_conv2d(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3), dtype=np.float32)
        binary_conv2d = BinaryConv2D(strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3), in_channel=3)
        y = binary_conv2d.construct(x, filters)
        self.assertEqual(y.shape, (1, 1, 1, 3))

    def test_dorefa_conv2d(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        filters = np.ones((3, 3, 3, 3), dtype=np.float32)
        dorefa_conv2d = DorefaConv2D(bitW=1, bitA=1, strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1, 1, 1), out_channel=3, k_size=(3, 3), in_channel=3)
        y = dorefa_conv2d.construct(x, filters)
        self.assertEqual(y.shape, (1, 1, 1, 3))

    def test_rnncell(self):
        input = np.ones((1, 3), dtype=np.float32)
        h = np.ones((1, 3), dtype=np.float32)
        weight_ih = np.ones((3, 3), dtype=np.float32)
        weight_hh = np.ones((3, 3), dtype=np.float32)
        bias_ih = np.ones((3,), dtype=np.float32)
        bias_hh = np.ones((3,), dtype=np.float32)
        rnn_cell = rnncell(weight_ih, weight_hh, bias_ih, bias_hh, act="relu")
        h_new, _ = rnn_cell.construct(input, h)
        self.assertEqual(h_new.shape, (1, 3))

    def test_lstmcell(self):
        input = np.ones((1, 3), dtype=np.float32)
        h = np.ones((1, 3), dtype=np.float32)
        c = np.ones((1, 3), dtype=np.float32)
        weight_ih = np.ones((3, 12), dtype=np.float32)
        weight_hh = np.ones((3, 12), dtype=np.float32)
        bias_ih = np.ones((12,), dtype=np.float32)
        bias_hh = np.ones((12,), dtype=np.float32)
        lstm_cell = lstmcell(weight_ih, weight_hh, bias_ih, bias_hh)
        h_new, _, c_new = lstm_cell.construct(input, h, c)
        self.assertEqual(h_new.shape, (1, 3))
        self.assertEqual(c_new.shape, (1, 3))

    def test_grucell(self):
        input = np.ones((1, 3), dtype=np.float32)
        h = np.ones((1, 3), dtype=np.float32)
        weight_ih = np.ones((3, 9), dtype=np.float32)
        weight_hh = np.ones((3, 9), dtype=np.float32)
        bias_ih = np.ones((9,), dtype=np.float32)
        bias_hh = np.ones((9,), dtype=np.float32)
        gru_cell = grucell(weight_ih, weight_hh, bias_ih, bias_hh)
        h_new, _ = gru_cell.construct(input, h)
        self.assertEqual(h_new.shape, (1, 3))

    def test_rnnbase(self):
        input = np.ones((5, 3, 10), dtype=np.float32)
        weight_ih = [np.ones((10, 20), dtype=np.float32) for _ in range(2)]
        weight_hh = [np.ones((10, 20), dtype=np.float32) for _ in range(2)]
        bias_ih = [np.ones((20,), dtype=np.float32) for _ in range(2)]
        bias_hh = [np.ones((20,), dtype=np.float32) for _ in range(2)]
        rnn_base = rnnbase(mode="LSTM", input_size=10, hidden_size=10, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False, is_train=True, w_ih=weight_ih, w_hh=weight_hh, b_ih=bias_ih, b_hh=bias_hh)
        output, (h_n, c_n) = rnn_base.construct(input)
        self.assertEqual(output.shape, (5, 3, 10))
        self.assertEqual(h_n.shape, (1, 3, 10))
        self.assertEqual(c_n.shape, (1, 3, 10))

    def test_layernorm(self):
        input = np.ones((2, 3, 4), dtype=np.float32)
        gamma = np.ones((4,), dtype=np.float32)
        beta = np.zeros((4,), dtype=np.float32)
        layer_norm = layernorm(normalized_shape=(4,), gamma=gamma, beta=beta, eps=1e-5, input_shape=(2, 3, 4))
        output = layer_norm.construct(input)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_multiheadattention(self):
        q = np.ones((5, 3, 10), dtype=np.float32)
        k = np.ones((5, 3, 10), dtype=np.float32)
        v = np.ones((5, 3, 10), dtype=np.float32)
        q_weight = np.ones((10, 10), dtype=np.float32)
        k_weight = np.ones((10, 10), dtype=np.float32)
        v_weight = np.ones((10, 10), dtype=np.float32)
        out_weight = np.ones((10, 10), dtype=np.float32)
        q_bias = np.zeros((10,), dtype=np.float32)
        k_bias = np.zeros((10,), dtype=np.float32)
        v_bias = np.zeros((10,), dtype=np.float32)
        out_bias = np.zeros((10,), dtype=np.float32)
        multihead_attention = multiheadattention(embed_dim=10, num_heads=2, dropout=0.0, batch_first=True, need_weights=False, q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, out_weight=out_weight, q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias, train=True)
        output, _ = multihead_attention.construct(q, k, v, None, None)
        self.assertEqual(output.shape, (5, 3, 10))

    def test_binary_dense(self):
        weights = np.ones((3, 3), dtype=np.float32)
        bias = np.ones((3,), dtype=np.float32)
        binary_dense = BinaryDense(weights, bias)
        with self.assertRaises(NotImplementedError):
            binary_dense.construct(np.ones((1, 3), dtype=np.float32))

    def test_dorefa_dense(self):
        weights = np.ones((3, 3), dtype=np.float32)
        bias = np.ones((3,), dtype=np.float32)
        dorefa_dense = DorefaDense(weights, bias, bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            dorefa_dense.construct(np.ones((1, 3), dtype=np.float32))

    def test_ternary_dense(self):
        weights = np.ones((3, 3), dtype=np.float32)
        bias = np.ones((3,), dtype=np.float32)
        ternary_dense = TernaryDense(weights, bias)
        with self.assertRaises(NotImplementedError):
            ternary_dense.construct(np.ones((1, 3), dtype=np.float32))

    def test_quan_dense(self):
        weights = np.ones((3, 3), dtype=np.float32)
        bias = np.ones((3,), dtype=np.float32)
        quan_dense = QuanDense(weights, bias, bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            quan_dense.construct(np.ones((1, 3), dtype=np.float32))

    def test_quan_dense_bn(self):
        weights = np.ones((3, 3), dtype=np.float32)
        scale_para = np.ones((3,), dtype=np.float32)
        offset_para = np.ones((3,), dtype=np.float32)
        moving_mean = np.ones((3,), dtype=np.float32)
        moving_variance = np.ones((3,), dtype=np.float32)
        quan_dense_bn = QuanDenseBn(weights, scale_para, offset_para, moving_mean, moving_variance, decay=0.9, bitW=1, bitA=1, epsilon=1e-5, is_train=True)
        with self.assertRaises(NotImplementedError):
            quan_dense_bn.construct(np.ones((1, 3), dtype=np.float32))

    def test_ternary_conv(self):
        weights = np.ones((3, 3, 3, 3), dtype=np.float32)
        ternary_conv = TernaryConv(weights, strides=(1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1))
        with self.assertRaises(NotImplementedError):
            ternary_conv.construct(np.ones((1, 3, 3, 3), dtype=np.float32))

    def test_quan_conv(self):
        weights = np.ones((3, 3, 3, 3), dtype=np.float32)
        quan_conv = QuanConv(weights, strides=(1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1), bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            quan_conv.construct(np.ones((1, 3, 3, 3), dtype=np.float32))

    def test_quan_conv_bn(self):
        weights = np.ones((3, 3, 3, 3), dtype=np.float32)
        scale_para = np.ones((3,), dtype=np.float32)
        offset_para = np.ones((3,), dtype=np.float32)
        moving_mean = np.ones((3,), dtype=np.float32)
        moving_variance = np.ones((3,), dtype=np.float32)
        quan_conv_bn = QuanConvBn(weights, scale_para, offset_para, moving_mean, moving_variance, strides=(1, 1), padding="VALID", data_format="NHWC", dilations=(1, 1), bitW=1, bitA=1, decay=0.9, epsilon=1e-5, is_train=True)
        with self.assertRaises(NotImplementedError):
            quan_conv_bn.construct(np.ones((1, 3, 3, 3), dtype=np.float32))

    def test_prelu(self):
        x = Tensor(np.ones((1, 3, 3, 3), dtype=np.float32))
        weight = np.ones((3,), dtype=np.float32)
        prelu_layer = PReLU(data_format="channels_last")
        y = prelu_layer(x, weight)
        self.assertEqual(y.shape, (1, 3, 3, 3))

    def test_prelu_function(self):
        x = Tensor(np.ones((1, 3, 3, 3), dtype=np.float32))
        weight = np.ones((3,), dtype=np.float32)
        y = prelu(x, weight, data_format="channels_last")
        self.assertEqual(y.shape, (1, 3, 3, 3))

    def test_hardsigmoid(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = hardsigmoid(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([0, 0.5, 1], dtype=np.float32), decimal=4)

    def test_hardswish(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = hardswish(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([0, 0, 0.6667], dtype=np.float32), decimal=4)

    def test_swish(self):
        x = Tensor(np.array([-1, 0, 1], dtype=np.float32))
        y = swish(x)
        np.testing.assert_array_almost_equal(y.asnumpy(), np.array([-0.2689, 0, 0.7311], dtype=np.float32), decimal=4)

    def test_linear(self):
        x = np.ones((2, 3), dtype=np.float32)
        weight = np.ones((3, 4), dtype=np.float32)
        bias = np.ones((4,), dtype=np.float32)
        y = linear(x, weight, bias)
        self.assertEqual(y.shape, (2, 4))

    def test_unfold(self):
        x = np.ones((1, 3, 3, 3), dtype=np.float32)
        y = unfold(x, kernel_size=(2, 2), dilation=1, padding=0, stride=1)
        self.assertEqual(y.shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
