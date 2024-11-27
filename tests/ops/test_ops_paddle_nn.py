import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "paddle"
import numpy as np

from tensorlayerx.backend.ops.paddle_nn import *
from tests.utils import CustomTestCase


class TestPaddleNN(CustomTestCase):

    def setUp(self):
        self.x = pd.to_tensor(np.random.rand(2, 3, 4, 5), dtype="float32")
        self.filters = pd.to_tensor(np.random.rand(3, 3, 3, 3), dtype="float32")
        self.bias = pd.to_tensor(np.random.rand(3), dtype="float32")

    def test_padding_format(self):
        self.assertEqual(padding_format("SAME"), "SAME")
        self.assertEqual(padding_format("VALID"), "VALID")
        self.assertEqual(padding_format("same"), "SAME")
        self.assertEqual(padding_format("valid"), "VALID")
        self.assertEqual(padding_format(None), None)
        self.assertEqual(padding_format((1, 1)), (1, 1))
        self.assertEqual(padding_format(1), 1)
        with self.assertRaises(Exception):
            padding_format("unsupported")

    def test_preprocess_1d_format(self):
        self.assertEqual(preprocess_1d_format("channels_last", "SAME"), ("NLC", "SAME"))
        self.assertEqual(preprocess_1d_format("channels_first", "VALID"), ("NCL", "VALID"))
        with self.assertRaises(Exception):
            preprocess_1d_format("unsupported", "SAME")

    def test_preprocess_2d_format(self):
        self.assertEqual(preprocess_2d_format("channels_last", "SAME"), ("NHWC", "SAME"))
        self.assertEqual(preprocess_2d_format("channels_first", "VALID"), ("NCHW", "VALID"))
        with self.assertRaises(Exception):
            preprocess_2d_format("unsupported", "SAME")

    def test_preprocess_3d_format(self):
        self.assertEqual(preprocess_3d_format("channels_last", "SAME"), ("NDHWC", "SAME"))
        self.assertEqual(preprocess_3d_format("channels_first", "VALID"), ("NCDHW", "VALID"))
        with self.assertRaises(Exception):
            preprocess_3d_format("unsupported", "SAME")

    def test_nchw_to_nhwc(self):
        x = pd.to_tensor(np.random.rand(2, 3, 4, 5), dtype="float32")
        expected_shape = (2, 4, 5, 3)
        self.assertEqual(nchw_to_nhwc(x).shape, expected_shape)

    def test_nhwc_to_nchw(self):
        x = pd.to_tensor(np.random.rand(2, 4, 5, 3), dtype="float32")
        expected_shape = (2, 3, 4, 5)
        self.assertEqual(nhwc_to_nchw(x).shape, expected_shape)

    def test_relu(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = np.maximum(x, 0)
        self.assertTrue(np.allclose(relu(pd.to_tensor(x)).numpy(), expected))

    def test_elu(self):
        x = np.random.rand(2, 3).astype("float32")
        alpha = 1.0
        expected = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        self.assertTrue(np.allclose(elu(pd.to_tensor(x)).numpy(), expected))

    def test_relu6(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = np.minimum(np.maximum(x, 0), 6)
        self.assertTrue(np.allclose(relu6(pd.to_tensor(x)).numpy(), expected))

    def test_leaky_relu(self):
        x = np.random.rand(2, 3).astype("float32")
        alpha = 0.01
        expected = np.where(x > 0, x, alpha * x)
        self.assertTrue(np.allclose(leaky_relu(pd.to_tensor(x)).numpy(), expected))

    def test_softplus(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = np.log(1 + np.exp(x))
        self.assertTrue(np.allclose(Softplus()(pd.to_tensor(x)).numpy(), expected))

    def test_tanh(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = np.tanh(x)
        self.assertTrue(np.allclose(Tanh()(pd.to_tensor(x)).numpy(), expected))

    def test_sigmoid(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = 1 / (1 + np.exp(-x))
        self.assertTrue(np.allclose(sigmoid(pd.to_tensor(x)).numpy(), expected))

    def test_softmax(self):
        x = np.random.rand(2, 3).astype("float32")
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.assertTrue(np.allclose(softmax(pd.to_tensor(x)).numpy(), expected))

    def test_gelu(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        self.assertTrue(np.allclose(gelu(pd.to_tensor(x)).numpy(), expected))

    def test_dropout(self):
        x = np.random.rand(2, 3).astype("float32")
        dropout_layer = Dropout(p=0.5, seed=1)
        output = dropout_layer(pd.to_tensor(x))
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(np.allclose(output.numpy().mean(), x.mean(), atol=0.1))

    def test_bias_add(self):
        x = np.random.rand(2, 3).astype("float32")
        bias = np.random.rand(3).astype("float32")
        expected = x + bias
        self.assertTrue(np.allclose(bias_add(pd.to_tensor(x), pd.to_tensor(bias)).numpy(), expected))

    def test_conv1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        filters = np.random.rand(3, 3, 3).astype("float32")
        # Expected shape calculation
        expected_shape = (2, 3, 4)
        self.assertEqual(conv1d(pd.to_tensor(x), pd.to_tensor(filters), 1, "SAME").shape, expected_shape)

    def test_conv2d(self):
        expected_shape = (2, 3, 4, 3)
        self.assertEqual(conv2d(self.x, self.filters, [1, 1, 1, 1], "SAME").shape, expected_shape)

    def test_conv3d(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        filters = np.random.rand(3, 3, 3, 3, 3).astype("float32")
        expected_shape = (2, 3, 4, 5, 3)
        self.assertEqual(conv3d(pd.to_tensor(x), pd.to_tensor(filters), [1, 1, 1, 1, 1], "SAME").shape, expected_shape)

    def test_max_pool1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        expected_shape = (2, 3, 2)
        self.assertEqual(max_pool1d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_max_pool2d(self):
        expected_shape = (2, 3, 2, 2)
        self.assertEqual(max_pool2d(self.x, 2).shape, expected_shape)

    def test_max_pool3d(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        expected_shape = (2, 3, 2, 2, 3)
        self.assertEqual(max_pool3d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_avg_pool1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        expected_shape = (2, 3, 2)
        self.assertEqual(avg_pool1d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_avg_pool2d(self):
        expected_shape = (2, 3, 2, 2)
        self.assertEqual(avg_pool2d(self.x, 2).shape, expected_shape)

    def test_avg_pool3d(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        expected_shape = (2, 3, 2, 2, 3)
        self.assertEqual(avg_pool3d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_depthwise_conv2d(self):
        x = np.random.rand(2, 4, 4, 3).astype("float32")
        filters = np.random.rand(3, 3, 3, 1).astype("float32")
        point_filters = np.random.rand(1, 1, 3, 3).astype("float32")
        expected_shape = (2, 4, 4, 3)
        depthwise_conv = DepthwiseConv2d(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", in_channels=3)
        self.assertEqual(depthwise_conv(pd.to_tensor(x), pd.to_tensor(filters), pd.to_tensor(point_filters)).shape, expected_shape)

    def test_conv1d_transpose(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        filters = np.random.rand(3, 3, 3).astype("float32")
        output_shape = [2, 3, 4]
        expected_shape = (2, 3, 4)
        self.assertEqual(conv1d_transpose(pd.to_tensor(x), pd.to_tensor(filters), output_shape, 1).shape, expected_shape)

    def test_conv2d_transpose(self):
        x = np.random.rand(2, 3, 4, 5).astype("float32")
        filters = np.random.rand(3, 3, 3, 3).astype("float32")
        output_shape = [2, 3, 4, 3]
        expected_shape = (2, 3, 4, 3)
        self.assertEqual(conv2d_transpose(pd.to_tensor(x), pd.to_tensor(filters), output_size=output_shape).shape, expected_shape)

    def test_conv3d_transpose(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        filters = np.random.rand(3, 3, 3, 3, 3).astype("float32")
        output_shape = [2, 3, 4, 5, 3]
        expected_shape = (2, 3, 4, 5, 3)
        self.assertEqual(conv3d_transpose(pd.to_tensor(x), pd.to_tensor(filters), output_shape, [1, 1, 1, 1, 1]).shape, expected_shape)

    def test_batch_norm(self):
        x = np.random.rand(2, 3, 4, 5).astype("float32")
        beta = np.random.rand(5).astype("float32")
        gamma = np.random.rand(5).astype("float32")
        moving_mean = np.random.rand(5).astype("float32")
        moving_var = np.random.rand(5).astype("float32")
        batch_norm = BatchNorm(beta=beta, gamma=gamma, moving_mean=moving_mean, moving_var=moving_var, num_features=5, is_train=True)
        self.assertEqual(batch_norm(pd.to_tensor(x)).shape, x.shape)

    def test_group_conv2d(self):
        x = np.random.rand(2, 3, 4, 6).astype("float32")
        filters = np.random.rand(3, 3, 3, 6).astype("float32")
        expected_shape = (2, 3, 4, 6)
        group_conv = GroupConv2D(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=6, k_size=[3, 3], groups=3)
        self.assertEqual(group_conv(pd.to_tensor(x), pd.to_tensor(filters)).shape, expected_shape)

    def test_separable_conv1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        depthwise_filters = np.random.rand(3, 3, 1).astype("float32")
        pointwise_filters = np.random.rand(1, 3, 3).astype("float32")
        expected_shape = (2, 3, 4)
        separable_conv = SeparableConv1D(stride=1, padding="SAME", data_format="NWC", dilations=1, out_channel=3, k_size=3, in_channel=3, depth_multiplier=1)
        self.assertEqual(separable_conv(pd.to_tensor(x), pd.to_tensor(depthwise_filters), pd.to_tensor(pointwise_filters)).shape, expected_shape)

    def test_separable_conv2d(self):
        x = np.random.rand(2, 3, 4, 5).astype("float32")
        depthwise_filters = np.random.rand(3, 3, 5, 1).astype("float32")
        pointwise_filters = np.random.rand(1, 1, 5, 5).astype("float32")
        expected_shape = (2, 3, 4, 5)
        separable_conv = SeparableConv2D(strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1], out_channel=5, k_size=[3, 3], in_channel=5, depth_multiplier=1)
        self.assertEqual(separable_conv(pd.to_tensor(x), pd.to_tensor(depthwise_filters), pd.to_tensor(pointwise_filters)).shape, expected_shape)

    def test_adaptive_avg_pool1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        expected_shape = (2, 3, 2)
        self.assertEqual(adaptive_avg_pool1d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_adaptive_avg_pool2d(self):
        expected_shape = (2, 3, 2, 2)
        self.assertEqual(adaptive_avg_pool2d(self.x, [2, 2]).shape, expected_shape)

    def test_adaptive_avg_pool3d(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        expected_shape = (2, 3, 2, 2, 2)
        self.assertEqual(adaptive_avg_pool3d(pd.to_tensor(x), [2, 2, 2]).shape, expected_shape)

    def test_adaptive_max_pool1d(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        expected_shape = (2, 3, 2)
        self.assertEqual(adaptive_max_pool1d(pd.to_tensor(x), 2).shape, expected_shape)

    def test_adaptive_max_pool2d(self):
        expected_shape = (2, 3, 2, 2)
        self.assertEqual(adaptive_max_pool2d(self.x, [2, 2]).shape, expected_shape)

    def test_adaptive_max_pool3d(self):
        x = np.random.rand(2, 3, 4, 5, 6).astype("float32")
        expected_shape = (2, 3, 2, 2, 2)
        self.assertEqual(adaptive_max_pool3d(pd.to_tensor(x), [2, 2, 2]).shape, expected_shape)

    def test_rnncell(self):
        input = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        h = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        weight_ih = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        weight_hh = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        bias_ih = pd.to_tensor(np.random.rand(3), dtype="float32")
        bias_hh = pd.to_tensor(np.random.rand(3), dtype="float32")
        cell = rnncell(weight_ih, weight_hh, bias_ih, bias_hh, act="relu")
        output, new_h = cell.forward(input, h)
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(new_h.shape, (2, 3))

    def test_lstmcell(self):
        input = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        h = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        c = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        weight_ih = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        weight_hh = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        bias_ih = pd.to_tensor(np.random.rand(3), dtype="float32")
        bias_hh = pd.to_tensor(np.random.rand(3), dtype="float32")
        cell = lstmcell(weight_ih, weight_hh, bias_ih, bias_hh)
        output, new_h, new_c = cell.forward(input, h, c)
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(new_h.shape, (2, 3))
        self.assertEqual(new_c.shape, (2, 3))

    def test_grucell(self):
        input = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        h = pd.to_tensor(np.random.rand(2, 3), dtype="float32")
        weight_ih = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        weight_hh = pd.to_tensor(np.random.rand(3, 3), dtype="float32")
        bias_ih = pd.to_tensor(np.random.rand(3), dtype="float32")
        bias_hh = pd.to_tensor(np.random.rand(3), dtype="float32")
        cell = grucell(weight_ih, weight_hh, bias_ih, bias_hh)
        output, new_h = cell.forward(input, h)
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(new_h.shape, (2, 3))

    def test_split_states(self):
        states = pd.to_tensor(np.random.rand(4, 2, 3), dtype="float32")
        split = split_states(states, bidirectional=True, state_components=1)
        self.assertEqual(len(split), 2)
        self.assertEqual(len(split[0]), 2)
        self.assertEqual(split[0][0].shape, (2, 3))

    def test_concat_states(self):
        states = [pd.to_tensor(np.random.rand(2, 3), dtype="float32") for _ in range(4)]
        concat = concat_states(states, bidirectional=True, state_components=1)
        self.assertEqual(concat.shape, (4, 2, 3))

    def test_layernorm(self):
        input = pd.to_tensor(np.random.rand(2, 3, 4), dtype="float32")
        gamma = pd.to_tensor(np.ones(4), dtype="float32")
        beta = pd.to_tensor(np.zeros(4), dtype="float32")
        norm = layernorm(normalized_shape=[4], gamma=gamma, beta=beta, eps=1e-5, input_shape=[2, 3, 4])
        output = norm(input)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_multiheadattention(self):
        q = pd.to_tensor(np.random.rand(2, 3, 4), dtype="float32")
        k = pd.to_tensor(np.random.rand(2, 3, 4), dtype="float32")
        v = pd.to_tensor(np.random.rand(2, 3, 4), dtype="float32")
        q_weight = pd.to_tensor(np.random.rand(4, 4), dtype="float32")
        k_weight = pd.to_tensor(np.random.rand(4, 4), dtype="float32")
        v_weight = pd.to_tensor(np.random.rand(4, 4), dtype="float32")
        out_weight = pd.to_tensor(np.random.rand(4, 4), dtype="float32")
        q_bias = pd.to_tensor(np.zeros(4), dtype="float32")
        k_bias = pd.to_tensor(np.zeros(4), dtype="float32")
        v_bias = pd.to_tensor(np.zeros(4), dtype="float32")
        out_bias = pd.to_tensor(np.zeros(4), dtype="float32")
        mha = multiheadattention(embed_dim=4, num_heads=2, dropout=0.1, batch_first=True, need_weights=False, q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, out_weight=out_weight, q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias, train=True)
        output, _ = mha(q, k, v, None, None)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_prelu(self):
        x = np.random.rand(2, 3, 4).astype("float32")
        weight = np.random.rand(3).astype("float32")
        expected = np.where(x > 0, x, weight[:, None, None] * x)
        self.assertTrue(np.allclose(prelu(pd.to_tensor(x), pd.to_tensor(weight), "NCL").numpy(), expected))

    def test_hardsigmoid(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = np.maximum(0, np.minimum(1, (x + 3) / 6))
        self.assertTrue(np.allclose(hardsigmoid(pd.to_tensor(x)).numpy(), expected))

    def test_hardswish(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = x * np.maximum(0, np.minimum(1, (x + 3) / 6))
        self.assertTrue(np.allclose(hardswish(pd.to_tensor(x)).numpy(), expected))

    def test_swish(self):
        x = np.random.rand(2, 3).astype("float32")
        expected = x / (1 + np.exp(-x))
        self.assertTrue(np.allclose(swish(pd.to_tensor(x)).numpy(), expected))

    def test_linear(self):
        x = np.random.rand(2, 3).astype("float32")
        weight = np.random.rand(3, 4).astype("float32")
        bias = np.random.rand(4).astype("float32")
        expected = np.dot(x, weight) + bias
        self.assertTrue(np.allclose(linear(pd.to_tensor(x), pd.to_tensor(weight), pd.to_tensor(bias)).numpy(), expected))

    def test_unfold(self):
        x = np.random.rand(2, 3, 4, 4).astype("float32")
        # Expected shape calculation
        expected_shape = (2, 12, 9)
        self.assertEqual(unfold(pd.to_tensor(x), 2).shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
