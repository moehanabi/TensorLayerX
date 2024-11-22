import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "oneflow"
import oneflow as flow

from tensorlayerx.backend.ops.oneflow_nn import *
from tests.utils import CustomTestCase


class TestOneFlowNN(CustomTestCase):

    def test_padding_format(self):
        self.assertEqual(padding_format("SAME"), "same")
        self.assertEqual(padding_format("same"), "same")
        self.assertEqual(padding_format("VALID"), "valid")
        self.assertEqual(padding_format("valid"), "valid")
        self.assertEqual(padding_format(None), None)
        self.assertEqual(padding_format((1, 2)), (1, 2))
        self.assertEqual(padding_format(1), 1)
        with self.assertRaises(Exception):
            padding_format("unsupported")

    def test_preprocess_1d_format(self):
        self.assertEqual(preprocess_1d_format("channels_last", "SAME"), ("NLC", "same"))
        self.assertEqual(preprocess_1d_format("NCW", "VALID"), ("NCL", "valid"))
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
        x = flow.randn(1, 3, 224, 224)
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 224, 224, 3))

    def test_nhwc_to_nchw(self):
        x = flow.randn(1, 224, 224, 3)
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 3, 224, 224))

    def test_relu(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = relu(x)
        expected = flow.tensor([0.0, 0.0, 1.0])
        self.assertTrue(flow.equal(y, expected))

    def test_elu(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = elu(x)
        expected = flow.tensor([-0.6321, 0.0, 1.0])  # Manually calculated expected values
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_relu6(self):
        x = flow.tensor([-1.0, 0.0, 6.0, 7.0])
        y = relu6(x)
        expected = flow.tensor([0.0, 0.0, 6.0, 6.0])
        self.assertTrue(flow.equal(y, expected))

    def test_leaky_relu(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = leaky_relu(x)
        expected = flow.tensor([-0.01, 0.0, 1.0])  # Assuming default negative slope of 0.01
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_sigmoid(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = sigmoid(x)
        expected = flow.tensor([0.2689, 0.5, 0.7311])  # Manually calculated expected values
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_softmax(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        y = softmax(x)
        expected = flow.tensor([0.0900, 0.2447, 0.6652])  # Manually calculated expected values
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_gelu(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = gelu(x)
        expected = flow.tensor([-0.1588, 0.0, 0.8413])  # Manually calculated expected values
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_bias_add(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        bias = flow.tensor([0.5])
        y = bias_add(x, bias)
        expected = flow.tensor([1.5, 2.5, 3.5])
        self.assertTrue(flow.equal(y, expected))

    def test_same_padding_1d(self):
        input = flow.randn(1, 3, 10)
        weight = flow.randn(2, 3, 3)
        strides = 1
        dilations = 1
        rows_odd, padding_rows = same_padding(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertEqual(padding_rows, 2)

    def test_same_padding_2d(self):
        input = flow.randn(1, 3, 32, 32)
        weight = flow.randn(3, 3, 3, 3)
        strides = (1, 1)
        dilations = (1, 1)
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertFalse(cols_odd)
        self.assertEqual(padding_rows, 2)
        self.assertEqual(padding_cols, 2)

    def test_same_padding_3d(self):
        input = flow.randn(1, 3, 16, 16, 16)
        weight = flow.randn(3, 3, 3, 3, 3)
        strides = (1, 1, 1)
        dilations = (1, 1, 1)
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertFalse(cols_odd)
        self.assertFalse(depth_odd)
        self.assertEqual(padding_rows, 2)
        self.assertEqual(padding_cols, 2)
        self.assertEqual(padding_depth, 2)

    def test_conv1d(self):
        x = flow.randn(1, 3, 10)
        filters = flow.randn(2, 3, 3)
        y = conv1d(x, filters, stride=1, padding="VALID")
        self.assertEqual(y.shape, (1, 2, 8))

    def test_conv2d(self):
        x = flow.randn(1, 3, 32, 32)
        filters = flow.randn(3, 3, 3, 3)
        y = conv2d(x, filters, strides=(1, 1), padding="VALID")
        self.assertEqual(y.shape, (1, 3, 30, 30))

    def test_conv3d(self):
        x = flow.randn(1, 3, 16, 16, 16)
        filters = flow.randn(3, 3, 3, 3, 3)
        y = conv3d(x, filters, strides=(1, 1, 1), padding="VALID")
        self.assertEqual(y.shape, (1, 3, 14, 14, 14))

    def test_local_response_norm(self):
        x = flow.randn(1, 3, 32, 32)
        y = local_response_norm(x, size=5)
        self.assertEqual(y.shape, x.shape)

    def test_lrn(self):
        x = flow.randn(1, 3, 32, 32)
        y = lrn(x, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
        self.assertEqual(y.shape, x.shape)

    def test_moments(self):
        x = flow.randn(1, 3, 32, 32)
        axes = [0, 2, 3]
        with self.assertRaises(NotImplementedError):
            mean, variance = moments(x, axes)

    def test_max_pool1d(self):
        x = flow.randn(1, 3, 10)
        pool_layer = MaxPool1d(ksize=2, strides=2, padding="VALID", return_mask=False)
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 5))

    def test_max_pool2d(self):
        x = flow.randn(1, 3, 32, 32)
        pool_layer = MaxPool(ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NCHW")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 16, 16))

    def test_max_pool3d(self):
        x = flow.randn(1, 3, 16, 16, 16)
        pool_layer = MaxPool3d(ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NCDHW")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 8, 8, 8))

    def test_avg_pool1d(self):
        x = flow.randn(1, 3, 10)
        pool_layer = AvgPool1d(ksize=2, strides=2, padding="VALID")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 5))

    def test_avg_pool2d(self):
        x = flow.randn(1, 3, 32, 32)
        pool_layer = AvgPool(ksize=2, strides=2, padding="VALID", data_format="NCHW")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 16, 16))

    def test_avg_pool3d(self):
        x = flow.randn(1, 3, 16, 16, 16)
        pool_layer = AvgPool3d(ksize=2, strides=2, padding="VALID", data_format="NCDHW")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 3, 8, 8, 8))

    def test_pool_max(self):
        x = flow.randn(1, 3, 32, 32)
        y = pool(x, window_shape=2, pooling_type="MAX", strides=2, padding="VALID", data_format="NCHW")
        self.assertEqual(y.shape, (1, 3, 16, 16))

    def test_pool_avg(self):
        x = flow.randn(1, 3, 32, 32)
        y = pool(x, window_shape=2, pooling_type="AVG", strides=2, padding="VALID", data_format="NCHW")
        self.assertEqual(y.shape, (1, 3, 16, 16))

    def test_depthwise_conv2d(self):
        x = flow.randn(1, 32, 32, 3)
        filters = flow.randn(3, 3, 3, 1)
        y = depthwise_conv2d(x, filters, strides=[1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1])
        self.assertEqual(y.shape, (1, 32, 32, 3))

    def test_same_padding_deconvolution_1d(self):
        input = flow.randn(1, 3, 10)
        weight = flow.randn(2, 3, 3)
        strides = 1
        dilations = 1
        rows_odd, padding_rows = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertEqual(padding_rows, 2)

    def test_same_padding_deconvolution_2d(self):
        input = flow.randn(1, 3, 32, 32)
        weight = flow.randn(3, 3, 3, 3)
        strides = (1, 1)
        dilations = (1, 1)
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertFalse(cols_odd)
        self.assertEqual(padding_rows, 2)
        self.assertEqual(padding_cols, 2)

    def test_same_padding_deconvolution_3d(self):
        input = flow.randn(1, 3, 16, 16, 16)
        weight = flow.randn(3, 3, 3, 3, 3)
        strides = (1, 1, 1)
        dilations = (1, 1, 1)
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertFalse(rows_odd)
        self.assertFalse(cols_odd)
        self.assertFalse(depth_odd)
        self.assertEqual(padding_rows, 2)
        self.assertEqual(padding_cols, 2)
        self.assertEqual(padding_depth, 2)

    def test_conv1d_transpose(self):
        x = flow.randn(1, 3, 10)
        filters = flow.randn(3, 3, 3)
        y = conv1d_transpose(x, filters, output_shape=[1, 3, 12], strides=1, padding="SAME", data_format="NWC")
        self.assertEqual(y.shape, (1, 12, 3))

    def test_conv2d_transpose(self):
        x = flow.randn(1, 3, 32, 32)
        filters = flow.randn(3, 3, 3, 3)
        y = conv2d_transpose(x, filters, output_shape=[1, 3, 34, 34], strides=(1, 1), padding="SAME", data_format="NHWC")
        self.assertEqual(y.shape, (1, 34, 34, 3))

    def test_conv3d_transpose(self):
        x = flow.randn(1, 3, 16, 16, 16)
        filters = flow.randn(3, 3, 3, 3, 3)
        y = conv3d_transpose(x, filters, output_shape=[1, 3, 18, 18, 18], strides=(1, 1, 1), padding="SAME", data_format="NDHWC")
        self.assertEqual(y.shape, (1, 18, 18, 18))

    def test_batch_normalization(self):
        x = flow.randn(1, 3, 32, 32)
        mean = flow.randn(3)
        variance = flow.randn(3)
        offset = flow.randn(3)
        scale = flow.randn(3)
        variance_epsilon = 1e-5
        with self.assertRaises(NotImplementedError):
            batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format="NCHW")

    def test_batch_norm(self):
        x = flow.randn(1, 3, 32, 32)
        bn_layer = BatchNorm(num_features=3, data_format="channels_first", is_train=True)
        y = bn_layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_group_conv2d(self):
        x = flow.randn(1, 3, 32, 32)
        filters = flow.randn(3, 1, 3, 3)
        group_conv_layer = GroupConv2D(strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1), out_channel=3, k_size=(3, 3), groups=3)
        y = group_conv_layer(x, filters)
        self.assertEqual(y.shape, (1, 3, 32, 32))

    def test_separable_conv1d(self):
        x = flow.randn(1, 3, 10)
        depthwise_filters = flow.randn(3, 1, 3)
        pointwise_filters = flow.randn(3, 3, 1)
        sep_conv_layer = SeparableConv1D(stride=1, padding="SAME", data_format="NLC", dilations=1, out_channel=3, k_size=3, in_channel=3, depth_multiplier=1)
        y = sep_conv_layer(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 10, 3))

    def test_separable_conv2d(self):
        x = flow.randn(1, 3, 32, 32)
        depthwise_filters = flow.randn(3, 1, 3, 3)
        pointwise_filters = flow.randn(3, 3, 1, 1)
        sep_conv_layer = SeparableConv2D(strides=(1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1), out_channel=3, k_size=(3, 3), in_channel=3, depth_multiplier=1)
        y = sep_conv_layer(x, depthwise_filters, pointwise_filters)
        self.assertEqual(y.shape, (1, 32, 32, 3))

    def test_adaptive_mean_pool1d(self):
        x = flow.randn(1, 3, 10)
        pool_layer = AdaptiveMeanPool1D(output_size=5, data_format="NLC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_mean_pool2d(self):
        x = flow.randn(1, 3, 32, 32)
        pool_layer = AdaptiveMeanPool2D(output_size=(16, 16), data_format="NHWC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_adaptive_mean_pool3d(self):
        x = flow.randn(1, 3, 16, 16, 16)
        pool_layer = AdaptiveMeanPool3D(output_size=(8, 8, 8), data_format="NDHWC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_adaptive_max_pool1d(self):
        x = flow.randn(1, 3, 10)
        pool_layer = AdaptiveMaxPool1D(output_size=5, data_format="NLC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 5, 3))

    def test_adaptive_max_pool2d(self):
        x = flow.randn(1, 3, 32, 32)
        pool_layer = AdaptiveMaxPool2D(output_size=(16, 16), data_format="NHWC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 16, 16, 3))

    def test_adaptive_max_pool3d(self):
        x = flow.randn(1, 3, 16, 16, 16)
        pool_layer = AdaptiveMaxPool3D(output_size=(8, 8, 8), data_format="NDHWC")
        y = pool_layer(x)
        self.assertEqual(y.shape, (1, 8, 8, 8, 3))

    def test_binary_conv2d(self):
        x = flow.randn(1, 3, 32, 32)
        filters = flow.randn(3, 3, 3, 3)
        binary_conv_layer = BinaryConv2D(strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1), out_channel=3, k_size=(3, 3), in_channel=3)
        with self.assertRaises(NotImplementedError):
            binary_conv_layer(x, filters)

    def test_dorefa_conv2d(self):
        x = flow.randn(1, 3, 32, 32)
        filters = flow.randn(3, 3, 3, 3)
        dorefa_conv_layer = DorefaConv2D(bitW=1, bitA=1, strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1), out_channel=3, k_size=(3, 3), in_channel=3)
        with self.assertRaises(NotImplementedError):
            dorefa_conv_layer(x, filters)

    def test_rnncell(self):
        x = flow.randn(1, 3)
        h = flow.randn(1, 3)
        weight_ih = flow.randn(3, 3)
        weight_hh = flow.randn(3, 3)
        bias_ih = flow.randn(3)
        bias_hh = flow.randn(3)
        rnn_cell = rnncell(weight_ih, weight_hh, bias_ih, bias_hh, act="tanh")
        y, new_h = rnn_cell(x, h)
        self.assertEqual(y.shape, (1, 3))
        self.assertEqual(new_h.shape, (1, 3))

    def test_lstmcell(self):
        x = flow.randn(1, 3)
        h = flow.randn(1, 3)
        c = flow.randn(1, 3)
        weight_ih = flow.randn(3, 3)
        weight_hh = flow.randn(3, 3)
        bias_ih = flow.randn(3)
        bias_hh = flow.randn(3)
        lstm_cell = lstmcell(weight_ih, weight_hh, bias_ih, bias_hh)
        y, new_h, new_c = lstm_cell(x, h, c)
        self.assertEqual(y.shape, (1, 3))
        self.assertEqual(new_h.shape, (1, 3))
        self.assertEqual(new_c.shape, (1, 3))

    def test_grucell(self):
        x = flow.randn(1, 3)
        h = flow.randn(1, 3)
        weight_ih = flow.randn(3, 3)
        weight_hh = flow.randn(3, 3)
        bias_ih = flow.randn(3)
        bias_hh = flow.randn(3)
        gru_cell = grucell(weight_ih, weight_hh, bias_ih, bias_hh)
        y, new_h = gru_cell(x, h)
        self.assertEqual(y.shape, (1, 3))
        self.assertEqual(new_h.shape, (1, 3))

    def test_rnnbase(self):
        x = flow.randn(5, 3, 10)
        weight_ih = [flow.randn(10, 10) for _ in range(2)]
        weight_hh = [flow.randn(10, 10) for _ in range(2)]
        bias_ih = [flow.randn(10) for _ in range(2)]
        bias_hh = [flow.randn(10) for _ in range(2)]
        rnn = rnnbase(
            mode="RNN_TANH",
            input_size=10,
            hidden_size=10,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
            is_train=True,
            w_ih=weight_ih,
            w_hh=weight_hh,
            b_ih=bias_ih,
            b_hh=bias_hh,
        )
        y, h = rnn(x, None)
        self.assertEqual(y.shape, (5, 3, 10))
        self.assertEqual(h.shape, (1, 3, 10))

    def test_layernorm(self):
        x = flow.randn(1, 3, 32, 32)
        gamma = flow.randn(32)
        beta = flow.randn(32)
        layer_norm_layer = layernorm(normalized_shape=(32,), gamma=gamma, beta=beta, eps=1e-5, input_shape=(1, 3, 32, 32))
        y = layer_norm_layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_multiheadattention(self):
        q = flow.randn(5, 3, 10)
        k = flow.randn(5, 3, 10)
        v = flow.randn(5, 3, 10)
        q_weight = flow.randn(10, 10)
        k_weight = flow.randn(10, 10)
        v_weight = flow.randn(10, 10)
        out_weight = flow.randn(10, 10)
        q_bias = flow.randn(10)
        k_bias = flow.randn(10)
        v_bias = flow.randn(10)
        out_bias = flow.randn(10)
        attn_layer = multiheadattention(
            embed_dim=10,
            num_heads=2,
            dropout=0.0,
            batch_first=False,
            need_weights=True,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            out_weight=out_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            out_bias=out_bias,
            train=True,
        )
        y, attn_weights = attn_layer(q, k, v, None, None)
        self.assertEqual(y.shape, (5, 3, 10))
        self.assertEqual(attn_weights.shape, (2, 5, 3, 3))

    def test_binary_dense(self):
        x = flow.randn(1, 10)
        weights = flow.randn(10, 10)
        bias = flow.randn(10)
        binary_dense_layer = BinaryDense(weights, bias)
        with self.assertRaises(NotImplementedError):
            binary_dense_layer(x)

    def test_dorefa_dense(self):
        x = flow.randn(1, 10)
        weights = flow.randn(10, 10)
        bias = flow.randn(10)
        dorefa_dense_layer = DorefaDense(weights, bias, bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            dorefa_dense_layer(x)

    def test_ternary_dense(self):
        x = flow.randn(1, 10)
        weights = flow.randn(10, 10)
        bias = flow.randn(10)
        ternary_dense_layer = TernaryDense(weights, bias)
        with self.assertRaises(NotImplementedError):
            ternary_dense_layer(x)

    def test_quan_dense(self):
        x = flow.randn(1, 10)
        weights = flow.randn(10, 10)
        bias = flow.randn(10)
        quan_dense_layer = QuanDense(weights, bias, bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            quan_dense_layer(x)

    def test_quan_dense_bn(self):
        x = flow.randn(1, 10)
        weights = flow.randn(10, 10)
        scale_para = flow.randn(10)
        offset_para = flow.randn(10)
        moving_mean = flow.randn(10)
        moving_variance = flow.randn(10)
        quan_dense_bn_layer = QuanDenseBn(weights, scale_para, offset_para, moving_mean, moving_variance, decay=0.9, bitW=1, bitA=1, epsilon=1e-5, is_train=True)
        with self.assertRaises(NotImplementedError):
            quan_dense_bn_layer(x)

    def test_ternary_conv(self):
        x = flow.randn(1, 3, 32, 32)
        weights = flow.randn(3, 3, 3, 3)
        ternary_conv_layer = TernaryConv(weights, strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1))
        with self.assertRaises(NotImplementedError):
            ternary_conv_layer(x)

    def test_quan_conv(self):
        x = flow.randn(1, 3, 32, 32)
        weights = flow.randn(3, 3, 3, 3)
        quan_conv_layer = QuanConv(weights, strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1), bitW=1, bitA=1)
        with self.assertRaises(NotImplementedError):
            quan_conv_layer(x)

    def test_quan_conv_bn(self):
        x = flow.randn(1, 3, 32, 32)
        weights = flow.randn(3, 3, 3, 3)
        scale_para = flow.randn(3)
        offset_para = flow.randn(3)
        moving_mean = flow.randn(3)
        moving_variance = flow.randn(3)
        quan_conv_bn_layer = QuanConvBn(weights, scale_para, offset_para, moving_mean, moving_variance, strides=(1, 1), padding="SAME", data_format="NCHW", dilations=(1, 1), bitW=1, bitA=1, decay=0.9, epsilon=1e-5, is_train=True)
        with self.assertRaises(NotImplementedError):
            quan_conv_bn_layer(x)

    def test_prelu(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        weight = flow.tensor([0.25])
        y = prelu(x, weight, data_format="NCHW")
        expected = flow.tensor([-0.25, 0.0, 1.0])
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_hardsigmoid(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = hardsigmoid(x)
        expected = flow.tensor([0.0, 0.5, 1.0])
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_hardswish(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = hardswish(x)
        expected = flow.tensor([0.0, 0.0, 0.6667])
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_swish(self):
        x = flow.tensor([-1.0, 0.0, 1.0])
        y = swish(x)
        expected = flow.tensor([-0.2689, 0.0, 0.7311])
        self.assertTrue(flow.allclose(y, expected, atol=1e-4))

    def test_linear(self):
        x = flow.tensor([[1.0, 2.0], [3.0, 4.0]])
        weight = flow.tensor([[1.0, 0.0], [0.0, 1.0]])
        bias = flow.tensor([1.0, 1.0])
        y = linear(x, weight, bias)
        expected = flow.tensor([[2.0, 3.0], [4.0, 5.0]])
        self.assertTrue(flow.equal(y, expected))

    def test_unfold(self):
        x = flow.randn(1, 3, 4, 4)
        y = unfold(x, kernel_size=3, dilation=1, padding=0, stride=1)
        self.assertEqual(y.shape, (1, 27, 4))


if __name__ == "__main__":
    unittest.main()
