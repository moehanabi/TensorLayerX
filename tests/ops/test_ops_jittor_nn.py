import unittest

import jittor as jt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "jittor"

from tensorlayerx.backend.ops.jittor_nn import *
from tests.utils import CustomTestCase


class TestJittorNN(CustomTestCase):

    def setUp(self):
        jt.flags.use_cuda = 1

    def test_padding_format(self):
        self.assertEqual(padding_format("SAME"), "same")
        self.assertEqual(padding_format("VALID"), "valid")
        self.assertEqual(padding_format("same"), "same")
        self.assertEqual(padding_format("valid"), "valid")
        self.assertEqual(padding_format((1, 2)), (1, 2))
        self.assertEqual(padding_format(1), 1)
        with self.assertRaises(Exception):
            padding_format("unsupported")

    def test_preprocess_padding(self):
        self.assertEqual(preprocess_padding("same", "2d"), "same")
        self.assertEqual(preprocess_padding((1, 2), "2d"), (1, 1, 2, 2))
        self.assertEqual(preprocess_padding(1, "2d"), 1)
        with self.assertRaises(RuntimeError):
            preprocess_padding("same", "4d")

    def test_preprocess_1d_format(self):
        self.assertEqual(preprocess_1d_format("channels_last", "same"), ("NLC", "same"))
        self.assertEqual(preprocess_1d_format("channels_first", "valid"), ("NCL", "valid"))
        with self.assertRaises(Exception):
            preprocess_1d_format("unsupported", "same")

    def test_preprocess_2d_format(self):
        self.assertEqual(preprocess_2d_format("channels_last", "same"), ("NHWC", "same"))
        self.assertEqual(preprocess_2d_format("channels_first", "valid"), ("NCHW", "valid"))
        with self.assertRaises(Exception):
            preprocess_2d_format("unsupported", "same")

    def test_preprocess_3d_format(self):
        self.assertEqual(preprocess_3d_format("channels_last", "same"), ("NDHWC", "same"))
        self.assertEqual(preprocess_3d_format("channels_first", "valid"), ("NCDHW", "valid"))
        with self.assertRaises(Exception):
            preprocess_3d_format("unsupported", "same")

    def test_nchw_to_nhwc(self):
        x = jt.array(np.random.randn(1, 3, 4, 4))
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 4, 4, 3))

    def test_nhwc_to_nchw(self):
        x = jt.array(np.random.randn(1, 4, 4, 3))
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 3, 4, 4))

    def test_relu(self):
        x = jt.array([-1.0, 0.0, 1.0])
        y = relu(x)
        np.testing.assert_array_equal(y.numpy(), [0.0, 0.0, 1.0])

    def test_elu(self):
        x = jt.array([-1.0, 0.0, 1.0])
        y = elu(x)
        np.testing.assert_array_almost_equal(y.numpy(), [-0.6321, 0.0, 1.0], decimal=4)

    def test_relu6(self):
        x = jt.array([-1.0, 0.0, 6.0, 7.0])
        y = relu6(x)
        np.testing.assert_array_equal(y.numpy(), [0.0, 0.0, 6.0, 6.0])

    def test_leaky_relu(self):
        x = jt.array([-1.0, 0.0, 1.0])
        y = leaky_relu(x)
        np.testing.assert_array_almost_equal(y.numpy(), [-0.01, 0.0, 1.0], decimal=4)

    def test_softplus(self):
        x = jt.array([-1.0, 0.0, 1.0])
        softplus = Softplus()
        y = softplus(x)
        np.testing.assert_array_almost_equal(y.numpy(), [0.3133, 0.6931, 1.3133], decimal=4)

    def test_tanh(self):
        x = jt.array([-1.0, 0.0, 1.0])
        tanh = Tanh()
        y = tanh(x)
        np.testing.assert_array_almost_equal(y.numpy(), [-0.7616, 0.0, 0.7616], decimal=4)

    def test_sigmoid(self):
        x = jt.array([-1.0, 0.0, 1.0])
        y = sigmoid(x)
        np.testing.assert_array_almost_equal(y.numpy(), [0.2689, 0.5, 0.7311], decimal=4)

    def test_softmax(self):
        x = jt.array([1.0, 2.0, 3.0])
        y = softmax(x)
        np.testing.assert_array_almost_equal(y.numpy(), [0.0900, 0.2447, 0.6652], decimal=4)

    def test_gelu(self):
        x = jt.array([-1.0, 0.0, 1.0])
        y = gelu(x)
        np.testing.assert_array_almost_equal(y.numpy(), [-0.1588, 0.0, 0.8413], decimal=4)

    def test_dropout(self):
        x = jt.ones((10, 10))
        y = dropout(x, p=0.5, is_train=True)
        self.assertTrue((y.numpy() == 0).sum() > 0)

    def test_bias_add(self):
        x = jt.ones((1, 3, 4, 4))
        bias = jt.array([1.0, 2.0, 3.0])
        y = bias_add(x, bias, data_format="channels_first")
        np.testing.assert_array_equal(y.numpy(), np.ones((1, 3, 4, 4)) + np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1))

    def test_conv1d(self):
        x = jt.ones((1, 3, 10))
        filters = jt.ones((2, 3, 3))
        y = conv1d(x, filters, stride=1, padding="valid")
        self.assertEqual(y.shape, (1, 2, 8))

    def test_conv2d(self):
        x = jt.ones((1, 3, 10, 10))
        filters = jt.ones((2, 3, 3, 3))
        y = conv2d(x, filters, strides=1, padding="valid")
        self.assertEqual(y.shape, (1, 2, 8, 8))

    def test_conv3d(self):
        x = jt.ones((1, 3, 10, 10, 10))
        filters = jt.ones((2, 3, 3, 3, 3))
        y = conv3d(x, filters, strides=1, padding="valid")
        self.assertEqual(y.shape, (1, 2, 8, 8, 8))

    def test_avg_pool1d(self):
        with self.assertRaises(NotImplementedError):
            avg_pool1d(jt.ones((1, 3, 10)), kernel_size=2, stride=2, padding=0)

    def test_avg_pool2d(self):
        x = jt.ones((1, 3, 10, 10))
        y = avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        self.assertEqual(y.shape, (1, 3, 5, 5))

    def test_avg_pool3d(self):
        x = jt.ones((1, 3, 10, 10, 10))
        y = avg_pool3d(x, kernel_size=2, stride=2, padding=0)
        self.assertEqual(y.shape, (1, 3, 5, 5, 5))

    def test_max_pool3d(self):
        x = jt.ones((1, 3, 10, 10, 10))
        y = max_pool3d(x, kernel_size=2, stride=2, padding=0)
        self.assertEqual(y.shape, (1, 3, 5, 5, 5))

    def test_pool_avg(self):
        x = jt.ones((1, 3, 10, 10))
        y = pool(x, window_shape=2, pooling_type="AVG", strides=2, padding="VALID")
        self.assertEqual(y.shape, (1, 3, 5, 5))

    def test_pool_max(self):
        x = jt.ones((1, 3, 10, 10))
        y = pool(x, window_shape=2, pooling_type="MAX", strides=2, padding="VALID")
        self.assertEqual(y.shape, (1, 3, 5, 5))

    def test_depthwise_conv2d(self):
        x = jt.ones((1, 10, 10, 3))
        filters = jt.ones((3, 3, 3, 1))
        y = depthwise_conv2d(x, filters, strides=[1, 1], padding="VALID", data_format="NHWC", dilations=[1, 1])
        self.assertEqual(y.shape, (1, 8, 8, 3))

    def test_same_padding_deconvolution(self):
        x = jt.ones((1, 3, 10, 10))
        filters = jt.ones((3, 3, 3, 3))
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding_deconvolution(x, filters, strides=[1, 1], dilations=[1, 1])
        self.assertFalse(rows_odd)
        self.assertFalse(cols_odd)
        self.assertEqual(padding_rows, 2)
        self.assertEqual(padding_cols, 2)

    def test_conv2d_transpose(self):
        x = jt.ones((1, 3, 10, 10))
        filters = jt.ones((3, 2, 3, 3))
        y = conv2d_transpose(x, filters, stride=1, padding=0)
        self.assertEqual(y.shape, (1, 2, 12, 12))

    def test_conv3d_transpose(self):
        x = jt.ones((1, 3, 10, 10, 10))
        filters = jt.ones((3, 2, 3, 3, 3))
        y = conv3d_transpose(x, filters, output_shape=[1, 2, 12, 12, 12], strides=[1, 1, 1], padding="SAME")
        self.assertEqual(y.shape, (1, 2, 12, 12, 12))

    def test_batch_norm(self):
        x = jt.ones((1, 3, 10, 10))
        bn = BatchNorm(num_features=3, is_train=True)
        y = bn(x)
        self.assertEqual(y.shape, (1, 3, 10, 10))

    def test_group_conv2d(self):
        x = jt.ones((1, 3, 10, 10))
        filters = jt.ones((2, 1, 3, 3))
        group_conv = GroupConv2D(strides=1, padding="VALID", data_format="NCHW", dilations=1, out_channel=2, k_size=3, groups=3)
        y = group_conv(x, filters)
        self.assertEqual(y.shape, (1, 2, 8, 8))

    def test_separable_conv1d(self):
        x = jt.ones((1, 10, 3))
        depthwise_filters = jt.ones((3, 1, 3))
        pointwise_filters = jt.ones((1, 3, 3))
        separable_conv = SeparableConv1D(stride=1, padding="VALID", data_format="NLC", dilations=1, out_channel=3, k_size=3, in_channel=3, depth_multiplier=1)
        y = separable_conv(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 8, 3))

    def test_separable_conv2d(self):
        x = jt.ones((1, 10, 10, 3))
        depthwise_filters = jt.ones((3, 3, 3, 1))
        pointwise_filters = jt.ones((1, 1, 3, 3))
        separable_conv = SeparableConv2D(strides=[1, 1], padding="VALID", data_format="NHWC", dilations=[1, 1], out_channel=3, k_size=3, in_channel=3, depth_multiplier=1)
        y = separable_conv(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 8, 8, 3))

    def test_adaptive_mean_pool1d(self):
        with self.assertRaises(NotImplementedError):
            AdaptiveMeanPool1D()(jt.ones((1, 10, 3)))

    def test_adaptive_mean_pool2d(self):
        with self.assertRaises(NotImplementedError):
            AdaptiveMeanPool2D()(jt.ones((1, 10, 10, 3)))

    def test_adaptive_mean_pool3d(self):
        with self.assertRaises(NotImplementedError):
            AdaptiveMeanPool3D()(jt.ones((1, 10, 10, 10, 3)))

    def test_adaptive_max_pool1d(self):
        with self.assertRaises(NotImplementedError):
            AdaptiveMaxPool1D()(jt.ones((1, 10, 3)))

    def test_adaptive_max_pool2d(self):
        x = jt.ones((1, 3, 10, 10))
        adaptive_max_pool = AdaptiveMaxPool2D(output_size=(5, 5), data_format="NCHW")
        y = adaptive_max_pool(x)
        self.assertEqual(y.shape, (1, 3, 5, 5))

    def test_adaptive_max_pool3d(self):
        with self.assertRaises(NotImplementedError):
            AdaptiveMaxPool3D(output_size=(5, 5, 5), data_format="NCDHW")(jt.ones((1, 3, 10, 10, 10)))

    def test_binary_conv2d(self):
        with self.assertRaises(NotImplementedError):
            BinaryConv2D()(jt.ones((1, 3, 10, 10)), jt.ones((3, 3, 3, 3)))

    def test_dorefa_conv2d(self):
        with self.assertRaises(NotImplementedError):
            DorefaConv2D()(jt.ones((1, 3, 10, 10)), jt.ones((3, 3, 3, 3)))

    def test_rnncell(self):
        weight_ih = jt.ones((3, 3))
        weight_hh = jt.ones((3, 3))
        bias_ih = jt.ones(3)
        bias_hh = jt.ones(3)
        rnn = rnncell(weight_ih, weight_hh, bias_ih, bias_hh, "tanh")
        input = jt.ones((2, 3))
        hx = jt.ones((2, 3))
        y, hy = rnn(input, hx)
        self.assertEqual(y.shape, (2, 3))
        self.assertEqual(hy.shape, (2, 3))

    def test_lstmcell(self):
        weight_ih = jt.ones((3, 12))
        weight_hh = jt.ones((3, 12))
        bias_ih = jt.ones(12)
        bias_hh = jt.ones(12)
        lstm = lstmcell(weight_ih, weight_hh, bias_ih, bias_hh)
        input = jt.ones((2, 3))
        h = jt.ones((2, 3))
        c = jt.ones((2, 3))
        h_new, hy, c_new = lstm(input, h, c)
        self.assertEqual(h_new.shape, (2, 3))
        self.assertEqual(hy.shape, (2, 3))
        self.assertEqual(c_new.shape, (2, 3))

    def test_grucell(self):
        weight_ih = jt.ones((3, 9))
        weight_hh = jt.ones((3, 9))
        bias_ih = jt.ones(9)
        bias_hh = jt.ones(9)
        gru = grucell(weight_ih, weight_hh, bias_ih, bias_hh)
        input = jt.ones((2, 3))
        hx = jt.ones((2, 3))
        hy, hy_new = gru(input, hx)
        self.assertEqual(hy.shape, (2, 3))
        self.assertEqual(hy_new.shape, (2, 3))

    # def test_rnnbase(self):
    #     input_size = 3
    #     hidden_size = 3
    #     num_layers = 2
    #     rnn = rnnbase("RNN_TANH", input_size, hidden_size, num_layers)
    #     input = jt.ones((5, 2, 3))
    #     hx = jt.ones((2, 2, 3))
    #     output, hidden_n = rnn(input, hx)
    #     self.assertEqual(output.shape, (5, 2, 3))
    #     self.assertEqual(hidden_n.shape, (2, 2, 3))

    #     rnn = rnnbase("LSTM", input_size, hidden_size, num_layers)
    #     input = jt.ones((5, 2, 3))
    #     hx = (jt.ones((2, 2, 3)), jt.ones((2, 2, 3)))
    #     output, hidden_n = rnn(input, hx)
    #     self.assertEqual(output.shape, (5, 2, 3))
    #     self.assertEqual(hidden_n[0].shape, (2, 2, 3))
    #     self.assertEqual(hidden_n[1].shape, (2, 2, 3))

    #     rnn = rnnbase("GRU", input_size, hidden_size, num_layers)
    #     input = jt.ones((5, 2, 3))
    #     hx = jt.ones((2, 2, 3))
    #     output, hidden_n = rnn(input, hx)
    #     self.assertEqual(output.shape, (5, 2, 3))
    #     self.assertEqual(hidden_n.shape, (2, 2, 3))

    def test_layernorm(self):
        x = jt.ones((2, 3, 4))
        gamma = jt.ones(4)
        beta = jt.zeros(4)
        ln = layernorm(normalized_shape=(4,), gamma=gamma, beta=beta, eps=1e-5, input_shape=(2, 3, 4))
        y = ln(x)
        self.assertEqual(y.shape, (2, 3, 4))

    def test_multiheadattention(self):
        embed_dim = 8
        num_heads = 2
        mha = multiheadattention(embed_dim=embed_dim, num_heads=num_heads)
        query = jt.ones((5, 2, embed_dim))
        key = jt.ones((5, 2, embed_dim))
        value = jt.ones((5, 2, embed_dim))
        attn_output, attn_weights = mha(query, key, value)
        self.assertEqual(attn_output.shape, (5, 2, embed_dim))
        self.assertEqual(attn_weights.shape, (2, 5, 5))

    def test_binary_dense(self):
        with self.assertRaises(NotImplementedError):
            BinaryDense()(jt.ones((2, 3)))

    def test_dorefa_dense(self):
        with self.assertRaises(NotImplementedError):
            DorefaDense()(jt.ones((2, 3)))

    def test_ternary_dense(self):
        with self.assertRaises(NotImplementedError):
            TernaryDense()(jt.ones((2, 3)))

    def test_quan_dense(self):
        with self.assertRaises(NotImplementedError):
            QuanDense()(jt.ones((2, 3)))

    def test_quan_dense_bn(self):
        with self.assertRaises(NotImplementedError):
            QuanDenseBn(weights=jt.ones((2, 3)), scale_para=jt.ones(3), offset_para=jt.zeros(3), moving_mean=jt.zeros(3), moving_variance=jt.ones(3), decay=0.9, bitW=8, bitA=8, epsilon=1e-5, is_train=True)(jt.ones((2, 3)))

    def test_ternary_conv(self):
        with self.assertRaises(NotImplementedError):
            TernaryConv(weights=jt.ones((3, 3, 3, 3)), strides=1, padding="VALID", data_format="NHWC", dilations=1)(jt.ones((1, 10, 10, 3)))

    def test_quan_conv(self):
        with self.assertRaises(NotImplementedError):
            QuanConv(weights=jt.ones((3, 3, 3, 3)), strides=1, padding="VALID", data_format="NHWC", dilations=1, bitW=8, bitA=8)(jt.ones((1, 10, 10, 3)))

    def test_quan_conv_bn(self):
        with self.assertRaises(NotImplementedError):
            QuanConvBn(weights=jt.ones((3, 3, 3, 3)), scale_para=jt.ones(3), offset_para=jt.zeros(3), moving_mean=jt.zeros(3), moving_variance=jt.ones(3), strides=1, padding="VALID", data_format="NHWC", dilations=1, bitW=8, bitA=8, decay=0.9, epsilon=1e-5, is_train=True)(jt.ones((1, 10, 10, 3)))

    def test_swish(self):
        with self.assertRaises(NotImplementedError):
            Swish()(jt.ones((2, 3)))

    def test_prelu(self):
        x = jt.ones((1, 3, 4, 4))
        weight = jt.ones(1)
        prelu_layer = PReLU(data_format="channels_first")
        y = prelu_layer(x, weight)
        self.assertEqual(y.shape, (1, 3, 4, 4))

    def test_prelu_function(self):
        x = jt.ones((1, 3, 4, 4))
        weight = jt.ones(1)
        y = prelu(x, weight, data_format="channels_first")
        self.assertEqual(y.shape, (1, 3, 4, 4))

    # def test_hardsigmoid(self):
    #     with self.assertRaises(NotImplementedError):
    #         hardsigmoid(jt.ones((2, 3)))

    # def test_hardswish(self):
    #     with self.assertRaises(NotImplementedError):
    #         hardswish(jt.ones((2, 3)))

    # def test_swish_function(self):
    #     with self.assertRaises(NotImplementedError):
    #         swish(jt.ones((2, 3)))

    def test_linear(self):
        x = jt.ones((2, 3))
        weight = jt.ones((3, 4))
        bias = jt.ones(4)
        y = linear(x, weight, bias)
        self.assertEqual(y.shape, (2, 4))

    def test_unfold(self):
        x = jt.ones((1, 3, 10, 10))
        y = unfold(x, kernel_size=3, dilation=1, padding=0, stride=1)
        self.assertEqual(y.shape, (1, 27, 64))


if __name__ == "__main__":
    unittest.main()
