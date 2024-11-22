import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TL_BACKEND'] = 'torch'

import numpy as np
import torch
import torch.nn.functional as F

import tensorlayerx as tlx
from tensorlayerx.backend.ops.torch_nn import *
from tests.utils import CustomTestCase


class TestPaddingFormat(CustomTestCase):

    def test_padding_format_same(self):
        self.assertEqual(padding_format("same"), "same")
        self.assertEqual(padding_format("SAME"), "same")

    def test_padding_format_valid(self):
        self.assertEqual(padding_format("valid"), "valid")
        self.assertEqual(padding_format("VALID"), "valid")

    def test_padding_format_none(self):
        self.assertIsNone(padding_format(None))

    def test_padding_format_tuple(self):
        self.assertEqual(padding_format((1, 2)), (1, 2))

    def test_padding_format_int(self):
        self.assertEqual(padding_format(1), 1)

    def test_padding_format_invalid(self):
        with self.assertRaises(ValueError):
            padding_format("invalid")


class TestPreprocessFormat(CustomTestCase):

    def test_preprocess_1d_format(self):
        self.assertEqual(preprocess_1d_format("channels_last", "same"), ("NLC", "same"))
        self.assertEqual(preprocess_1d_format("NWC", "valid"), ("NLC", "valid"))
        self.assertEqual(preprocess_1d_format("NCW", "SAME"), ("NCL", "same"))
        self.assertEqual(preprocess_1d_format("channels_first", "VALID"), ("NCL", "valid"))
        self.assertEqual(preprocess_1d_format(None, "same"), (None, "same"))
        with self.assertRaises(Exception):
            preprocess_1d_format("unsupported_format", "same")
        with self.assertRaises(Exception):
            preprocess_1d_format("NWC", "unsupported_padding")

    def test_preprocess_2d_format(self):
        self.assertEqual(preprocess_2d_format("channels_last", "same"), ("NHWC", "same"))
        self.assertEqual(preprocess_2d_format("NHWC", "valid"), ("NHWC", "valid"))
        self.assertEqual(preprocess_2d_format("NCHW", "SAME"), ("NCHW", "same"))
        self.assertEqual(preprocess_2d_format("channels_first", "VALID"), ("NCHW", "valid"))
        self.assertEqual(preprocess_2d_format(None, "same"), (None, "same"))
        with self.assertRaises(Exception):
            preprocess_2d_format("unsupported_format", "same")
        with self.assertRaises(Exception):
            preprocess_2d_format("NHWC", "unsupported_padding")

    def test_preprocess_3d_format(self):
        self.assertEqual(preprocess_3d_format("channels_last", "same"), ("NDHWC", "same"))
        self.assertEqual(preprocess_3d_format("NDHWC", "valid"), ("NDHWC", "valid"))
        self.assertEqual(preprocess_3d_format("NCDHW", "SAME"), ("NCDHW", "same"))
        self.assertEqual(preprocess_3d_format("channels_first", "VALID"), ("NCDHW", "valid"))
        self.assertEqual(preprocess_3d_format(None, "same"), (None, "same"))
        with self.assertRaises(Exception):
            preprocess_3d_format("unsupported_format", "same")
        with self.assertRaises(Exception):
            preprocess_3d_format("NDHWC", "unsupported_padding")


class TestDataFormatConversion(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_nchw_to_nhwc(self):
        # Test with 4D tensor
        x = torch.randn(1, 3, 224, 224)
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 224, 224, 3))

        # Test with 5D tensor
        x = torch.randn(1, 3, 16, 224, 224)
        y = nchw_to_nhwc(x)
        self.assertEqual(y.shape, (1, 16, 224, 224, 3))

    def test_nhwc_to_nchw(self):
        # Test with 4D tensor
        x = torch.randn(1, 224, 224, 3)
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 3, 224, 224))

        # Test with 5D tensor
        x = torch.randn(1, 16, 224, 224, 3)
        y = nhwc_to_nchw(x)
        self.assertEqual(y.shape, (1, 3, 16, 224, 224))


class TestActivationFunctions(CustomTestCase):
    ## TODO: optimize the test cases

    def setUp(self):
        self.x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    def test_relu_function(self):
        result = relu(self.x)
        expected = F.relu(self.x)
        self.assertTrue(torch.equal(result, expected))

    def test_elu_function(self):
        result = elu(self.x, alpha=1.0)
        expected = F.elu(self.x, alpha=1.0)
        self.assertTrue(torch.equal(result, expected))

    def test_relu6_function(self):
        result = relu6(self.x)
        expected = F.relu6(self.x)
        self.assertTrue(torch.equal(result, expected))

    def test_leaky_relu_function(self):
        result = leaky_relu(self.x, negative_slope=0.01)
        expected = F.leaky_relu(self.x, negative_slope=0.01)
        self.assertTrue(torch.equal(result, expected))

    def test_softplus_class(self):
        softplus_obj = Softplus()
        result = softplus_obj(self.x)
        expected = F.softplus(self.x)
        self.assertTrue(torch.equal(result, expected))

    def test_tanh_class(self):
        tanh_obj = Tanh()
        result = tanh_obj(self.x)
        expected = torch.tanh(self.x)
        self.assertTrue(torch.equal(result, expected))

    def test_sigmoid_function(self):
        result = sigmoid(self.x)
        expected = F.sigmoid(self.x)
        self.assertTrue(torch.equal(result, expected))

    def test_softmax_function(self):
        result = softmax(self.x, axis=-1)
        expected = F.softmax(self.x, dim=-1)
        self.assertTrue(torch.equal(result, expected))

    def test_gelu_function(self):
        result = gelu(self.x, approximate=False)
        expected = F.gelu(self.x)
        self.assertTrue(torch.equal(result, expected))


class TestDropout(unittest.TestCase):

    def setUp(self):
        self.input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.p = 0.5
        self.seed = 42

    def test_dropout(self):
        dropout_layer = Dropout(p=self.p, seed=self.seed)
        output_tensor = dropout_layer(self.input_tensor)
        self.assertEqual(output_tensor.shape, self.input_tensor.shape)
        self.assertTrue(torch.all((output_tensor == 0) | (output_tensor == self.input_tensor / (1 - self.p))))

    def test_dropout_zero_probability(self):
        dropout_layer = Dropout(p=0.0, seed=self.seed)
        output_tensor = dropout_layer(self.input_tensor)
        self.assertTrue(torch.equal(output_tensor, self.input_tensor))

    def test_dropout_one_probability(self):
        dropout_layer = Dropout(p=1.0, seed=self.seed)
        output_tensor = dropout_layer(self.input_tensor)
        self.assertTrue(torch.all(output_tensor == 0))


class TestBiasAdd(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_bias_add_channels_last(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        bias = torch.tensor([1.0, 2.0])
        result = bias_add(x, bias, data_format="channels_last")
        expected = torch.tensor([[[2.0, 4.0], [4.0, 6.0]], [[6.0, 8.0], [8.0, 10.0]]])
        self.assertTrue(torch.equal(result, expected))

    def test_bias_add_channels_first(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        bias = torch.tensor([1.0, 2.0])
        result = bias_add(x, bias, data_format="channels_first")
        expected = torch.tensor([[[2.0, 3.0], [4.0, 5.0]], [[7.0, 8.0], [9.0, 10.0]]])
        self.assertTrue(torch.equal(result, expected))

    def test_bias_add_default_format(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        bias = torch.tensor([1.0])
        result = bias_add(x, bias)
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))


class TestConv(CustomTestCase):

    def test_conv1d(self):
        input = torch.randn(1, 10, 3)  # [batch, in_width, in_channels]
        filters = torch.randn(2, 3, 3)  # [out_channels, in_channels, kernel_size]
        output = conv1d(input, filters, stride=1, padding="SAME")
        self.assertEqual(output.shape, (1, 10, 2))

    def test_conv2d(self):
        input = torch.randn(1, 10, 10, 3)  # [batch, in_height, in_width, in_channels]
        filters = torch.randn(3, 3, 3, 2)  # [filter_height, filter_width, in_channels, out_channels]
        output = conv2d(input, filters, strides=[1, 1, 1, 1], padding="SAME")
        self.assertEqual(output.shape, (1, 10, 10, 2))

    def test_conv3d(self):
        input = torch.randn(1, 10, 10, 10, 3)  # [batch, in_depth, in_height, in_width, in_channels]
        filters = torch.randn(3, 3, 3, 3, 2)  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        output = conv3d(input, filters, strides=[1, 1, 1, 1, 1], padding="SAME")
        self.assertEqual(output.shape, (1, 10, 10, 10, 2))


class TestLRN(CustomTestCase):
    def test_lrn(self):
        inputs = torch.randn(1, 3, 24, 24, dtype=torch.float32)
        depth_radius = 5
        bias = 1.0
        alpha = 1.0
        beta = 0.5

        output = lrn(inputs, depth_radius, bias, alpha, beta)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.all(output >= 0))


class TestMoments(CustomTestCase):
    def test_moments(self):
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
        axes = [0, 2, 3]

        with self.assertRaises(NotImplementedError):
            mean, variance = moments(x, axes)


class TestPoolingFunctions(CustomTestCase):
    def test_max_pool_1d(self):
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        expected_output = torch.tensor([[[2.0, 4.0]]])
        output = max_pool(input_tensor, ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_max_pool_2d(self):
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        expected_output = torch.tensor([[[[4.0]], [[8.0]]]])
        output = max_pool(input_tensor, ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NHWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_max_pool_3d(self):
        input_tensor = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
        expected_output = torch.tensor([[[[[8.0]]]]])
        output = max_pool(input_tensor, ksize=2, strides=2, padding="VALID", return_mask=False, data_format="NDHWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_max_pool_1d_same_padding(self):
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        expected_output = torch.tensor([[[2.0, 4.0, 5.0]]])
        output = max_pool(input_tensor, ksize=2, strides=2, padding="SAME", return_mask=False, data_format="NWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_max_pool_2d_same_padding(self):
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        expected_output = torch.tensor([[[[4.0, 4.0], [8.0, 8.0]]]])
        output = max_pool(input_tensor, ksize=2, strides=1, padding="SAME", return_mask=False, data_format="NHWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_max_pool_3d_same_padding(self):
        input_tensor = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
        expected_output = torch.tensor([[[[[8.0, 8.0], [8.0, 8.0]], [[8.0, 8.0], [8.0, 8.0]]]]])
        output = max_pool(input_tensor, ksize=2, strides=1, padding="SAME", return_mask=False, data_format="NDHWC")
        self.assertTrue(torch.equal(output, expected_output))

    def test_avg_pool_2d(self):
        input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
        ksize = [2, 2]
        strides = [2, 2]
        padding = "VALID"
        expected_output = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])

        output = avg_pool(input_data, ksize, strides, padding)
        np.testing.assert_almost_equal(output.numpy(), expected_output, decimal=5)

    def test_avg_pool_2d_same_padding(self):
        input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
        ksize = [2, 2]
        strides = [2, 2]
        padding = "SAME"
        expected_output = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])

        output = avg_pool(input_data, ksize, strides, padding)
        np.testing.assert_almost_equal(output.numpy(), expected_output, decimal=5)

    def test_avg_pool_3d(self):
        input_data = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
        ksize = [2, 2, 2]
        strides = [2, 2, 2]
        padding = "VALID"
        expected_output = np.array([[[[[4.0]]]]])

        output = avg_pool(input_data, ksize, strides, padding)
        np.testing.assert_almost_equal(output.numpy(), expected_output, decimal=5)


class TestDepthwiseConv2D(CustomTestCase):

    def test_depthwise_conv2d(self):
        # Define input tensor
        input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.float32)

        # Define filter tensor
        filter_tensor = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32)

        # Expected output tensor
        expected_output = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[6.0, 8.0, 10.0], [14.0, 16.0, 18.0], [22.0, 24.0, 26.0]]]], dtype=torch.float32)

        # Perform depthwise convolution
        output_tensor = depthwise_conv2d(input_tensor, filter_tensor, strides=[1, 1], padding="VALID", data_format="NHWC")

        # Check if the output is as expected
        np.testing.assert_allclose(output_tensor.detach().numpy(), expected_output.numpy(), rtol=1e-5)


class TestSamePaddingDeconvolution(unittest.TestCase):

    def test_same_padding_deconvolution_3d(self):
        input = torch.randn(1, 3, 10)
        weight = torch.randn(3, 3, 5)
        strides = 2
        dilations = 1
        rows_odd, padding_rows = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertEqual(rows_odd, False)
        self.assertEqual(padding_rows, 4)

    def test_same_padding_deconvolution_4d(self):
        input = torch.randn(1, 3, 10, 10)
        weight = torch.randn(3, 3, 5, 5)
        strides = (2, 2)
        dilations = (1, 1)
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertEqual(rows_odd, False)
        self.assertEqual(cols_odd, False)
        self.assertEqual(padding_rows, 4)
        self.assertEqual(padding_cols, 4)

    def test_same_padding_deconvolution_5d(self):
        input = torch.randn(1, 3, 10, 10, 10)
        weight = torch.randn(3, 3, 5, 5, 5)
        strides = (2, 2, 2)
        dilations = (1, 1, 1)
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding_deconvolution(input, weight, strides, dilations)
        self.assertEqual(rows_odd, False)
        self.assertEqual(cols_odd, False)
        self.assertEqual(depth_odd, False)
        self.assertEqual(padding_rows, 4)
        self.assertEqual(padding_cols, 4)
        self.assertEqual(padding_depth, 4)


class TestConvTranspose(CustomTestCase):

    def test_conv1d_transpose(self):
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        filters = torch.tensor([[[1.0, 0.5, 0.25]]], dtype=torch.float32)
        output_shape = [1, 1, 5]
        strides = 1
        padding = "SAME"
        data_format = "NWC"
        dilations = 1

        expected_output = torch.tensor([[[1.0, 2.5, 4.25, 2.0, 0.75]]], dtype=torch.float32)

        output = conv1d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        np.testing.assert_allclose(output.detach().numpy(), expected_output.numpy(), rtol=1e-5)

    def test_conv1d_transpose_valid_padding(self):
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        filters = torch.tensor([[[1.0, 0.5, 0.25]]], dtype=torch.float32)
        output_shape = [1, 1, 5]
        strides = 1
        padding = "VALID"
        data_format = "NWC"
        dilations = 1

        expected_output = torch.tensor([[[1.0, 2.5, 4.25, 2.0, 0.75]]], dtype=torch.float32)

        output = conv1d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        np.testing.assert_allclose(output.detach().numpy(), expected_output.numpy(), rtol=1e-5)

    def test_conv1d_transpose_dilations(self):
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        filters = torch.tensor([[[1.0, 0.5, 0.25]]], dtype=torch.float32)
        output_shape = [1, 1, 7]
        strides = 1
        padding = "SAME"
        data_format = "NWC"
        dilations = 2

        expected_output = torch.tensor([[[1.0, 0.0, 2.5, 0.0, 4.25, 0.0, 0.75]]], dtype=torch.float32)

        output = conv1d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        np.testing.assert_allclose(output.detach().numpy(), expected_output.numpy(), rtol=1e-5)

    def test_conv2d_transpose(self):
        # Define input tensor
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

        # Define filters
        filters = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=torch.float32)

        # Define expected output
        expected_output = torch.tensor([[[[1.0, 2.0, 0.0], [3.0, 3.0, -2.0], [0.0, -3.0, -4.0]]]], dtype=torch.float32)

        # Perform conv2d_transpose
        output = conv2d_transpose(input_tensor, filters, output_shape=None, strides=(1, 1), padding=0, data_format="NCHW")

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_conv2d_transpose_with_padding(self):
        # Define input tensor
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

        # Define filters
        filters = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=torch.float32)

        # Define expected output
        expected_output = torch.tensor([[[[1.0, 2.0, 0.0], [3.0, 3.0, -2.0], [0.0, -3.0, -4.0]]]], dtype=torch.float32)

        # Perform conv2d_transpose
        output = conv2d_transpose(input_tensor, filters, output_shape=None, strides=(1, 1), padding=1, data_format="NCHW")

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_conv2d_transpose_with_stride(self):
        # Define input tensor
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

        # Define filters
        filters = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=torch.float32)

        # Define expected output
        expected_output = torch.tensor([[[[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)

        # Perform conv2d_transpose
        output = conv2d_transpose(input_tensor, filters, output_shape=None, strides=(2, 2), padding=0, data_format="NCHW")

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_conv3d_transpose_valid_padding(self):
        input_tensor = torch.randn(1, 3, 4, 4, 4)
        filters = torch.randn(3, 3, 3, 3, 3)
        output_shape = [1, 3, 8, 8, 8]
        strides = [2, 2, 2]
        padding = "VALID"
        data_format = "NDHWC"
        dilations = [1, 1, 1]

        output = conv3d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        expected_output_shape = (1, 8, 8, 8, 3)
        self.assertEqual(output.shape, expected_output_shape)

    def test_conv3d_transpose_same_padding(self):
        input_tensor = torch.randn(1, 3, 4, 4, 4)
        filters = torch.randn(3, 3, 3, 3, 3)
        output_shape = [1, 3, 8, 8, 8]
        strides = [2, 2, 2]
        padding = "SAME"
        data_format = "NDHWC"
        dilations = [1, 1, 1]

        output = conv3d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        expected_output_shape = (1, 8, 8, 8, 3)
        self.assertEqual(output.shape, expected_output_shape)

    def test_conv3d_transpose_output_values(self):
        input_tensor = torch.ones(1, 3, 4, 4, 4)
        filters = torch.ones(3, 3, 3, 3, 3)
        output_shape = [1, 3, 8, 8, 8]
        strides = [2, 2, 2]
        padding = "SAME"
        data_format = "NDHWC"
        dilations = [1, 1, 1]

        output = conv3d_transpose(input_tensor, filters, output_shape, strides, padding, data_format, dilations)
        expected_output = np.ones((1, 8, 8, 8, 3)) * 27  # Since all ones, the output should be 27 for each element
        np.testing.assert_array_almost_equal(output.detach().numpy(), expected_output, decimal=5)


class TestGroupConv2D(unittest.TestCase):

    def setUp(self):
        self.input_data = torch.tensor(np.random.randn(1, 6, 5, 5), dtype=torch.float32)  # Batch size 1, 6 channels, 5x5 image
        self.filters = torch.tensor(np.random.randn(6, 1, 3, 3), dtype=torch.float32)  # 6 filters, 1 channel per group, 3x3 kernel
        self.strides = (1, 1)
        self.padding = "valid"
        self.data_format = "NCHW"
        self.dilations = (1, 1)
        self.out_channel = 6
        self.k_size = (3, 3)
        self.groups = 6

    def test_group_conv2d(self):
        group_conv2d = GroupConv2D(strides=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilations, out_channel=self.out_channel, k_size=self.k_size, groups=self.groups)
        output = group_conv2d(self.input_data, self.filters)
        expected_output_shape = (1, 6, 3, 3)  # Expected output shape after convolution
        self.assertEqual(output.shape, expected_output_shape)


class TestSeparableConv1D(unittest.TestCase):

    def setUp(self):
        self.input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32)
        self.depthwise_filter = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float32)
        self.pointwise_filter = torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float32)
        self.stride = 1
        self.padding = "valid"
        self.data_format = "NWC"
        self.dilations = 1
        self.out_channel = 1
        self.k_size = 3
        self.in_channel = 1
        self.depth_multiplier = 1

    def test_separable_conv1d(self):
        separable_conv1d = SeparableConv1D(stride=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilations, out_channel=self.out_channel, k_size=self.k_size, in_channel=self.in_channel, depth_multiplier=self.depth_multiplier)
        output = separable_conv1d(self.input, self.depthwise_filter, self.pointwise_filter)
        expected_output = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32)
        np.testing.assert_allclose(output.detach().numpy(), expected_output.numpy(), rtol=1e-5)


class TestSeparableConv2D(unittest.TestCase):

    def setUp(self):
        self.input = torch.tensor(np.random.randn(1, 3, 32, 32), dtype=torch.float32)
        self.depthwise_filter = torch.tensor(np.random.randn(3, 1, 3, 3), dtype=torch.float32)
        self.pointwise_filter = torch.tensor(np.random.randn(3, 3, 1, 1), dtype=torch.float32)
        self.strides = (1, 1)
        self.padding = "same"
        self.data_format = "NCHW"
        self.dilations = (1, 1)
        self.out_channel = 3
        self.k_size = (3, 3)
        self.in_channel = 3
        self.depth_multiplier = 1

    def test_separable_conv2d(self):
        separable_conv2d = SeparableConv2D(strides=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilations, out_channel=self.out_channel, k_size=self.k_size, in_channel=self.in_channel, depth_multiplier=self.depth_multiplier)
        output = separable_conv2d(self.input, self.depthwise_filter, self.pointwise_filter)
        self.assertEqual(output.shape, (1, 3, 32, 32))


class TestPooling(CustomTestCase):

    def test_adaptive_avg_pool1d(self):
        input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        output_size = 3
        expected_output = F.adaptive_avg_pool1d(input, output_size)
        output = adaptive_avg_pool1d(input, output_size)
        self.assertTrue(torch.equal(output, expected_output))

    def test_adaptive_avg_pool2d(self):
        input = torch.randn(1, 1, 4, 4)
        output_size = (2, 2)
        expected_output = F.adaptive_avg_pool2d(input, output_size)
        output = adaptive_avg_pool2d(input, output_size)
        self.assertTrue(torch.equal(output, expected_output))

    def test_adaptive_avg_pool3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        output_size = (2, 2, 2)
        expected_output = F.adaptive_avg_pool3d(input, output_size)
        output = adaptive_avg_pool3d(input, output_size)
        self.assertTrue(torch.equal(output, expected_output))

    def test_adaptive_max_pool1d(self):
        input = torch.tensor([[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]])
        output_size = 3
        expected_output, expected_indices = F.adaptive_max_pool1d(input, output_size, return_indices=True)
        output, indices = adaptive_max_pool1d(input, output_size, return_indices=True)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(indices, expected_indices))

    def test_adaptive_max_pool2d(self):
        input = torch.randn(1, 1, 4, 4)
        output_size = (2, 2)
        expected_output, expected_indices = F.adaptive_max_pool2d(input, output_size, return_indices=True)
        output, indices = adaptive_max_pool2d(input, output_size, return_indices=True)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(indices, expected_indices))

    def test_adaptive_max_pool3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        output_size = (2, 2, 2)
        expected_output, expected_indices = F.adaptive_max_pool3d(input, output_size, return_indices=True)
        output, indices = adaptive_max_pool3d(input, output_size, return_indices=True)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(indices, expected_indices))


class TestRNNCells(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.batch_size = 5

        self.input = torch.randn(self.batch_size, self.input_size)
        self.h = torch.randn(self.batch_size, self.hidden_size)
        self.c = torch.randn(self.batch_size, self.hidden_size)

        self.weight_ih = torch.randn(self.hidden_size, self.input_size)
        self.weight_hh = torch.randn(self.hidden_size, self.hidden_size)
        self.bias_ih = torch.randn(self.hidden_size)
        self.bias_hh = torch.randn(self.hidden_size)

    def test_rnncell_tanh(self):
        cell = rnncell(self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, act="tanh")
        output, h_new = cell(self.input, self.h)
        expected_output = torch.tanh(torch.nn.functional.linear(self.input, self.weight_ih, self.bias_ih) + torch.nn.functional.linear(self.h, self.weight_hh, self.bias_hh))
        self.assertTrue(torch.allclose(h_new, expected_output))

    def test_rnncell_relu(self):
        cell = rnncell(self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, act="relu")
        output, h_new = cell(self.input, self.h)
        expected_output = torch.relu(torch.nn.functional.linear(self.input, self.weight_ih, self.bias_ih) + torch.nn.functional.linear(self.h, self.weight_hh, self.bias_hh))
        self.assertTrue(torch.allclose(h_new, expected_output))

    def test_lstmcell(self):
        cell = lstmcell(self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        output, h_new, c_new = cell(self.input, self.h, self.c)
        self.assertEqual(h_new.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(c_new.shape, (self.batch_size, self.hidden_size))

    def test_grucell(self):
        cell = grucell(self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        output, h_new = cell(self.input, self.h)
        self.assertEqual(h_new.shape, (self.batch_size, self.hidden_size))


class TestRNNBase(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.num_layers = 2
        self.batch_size = 5
        self.seq_len = 7
        self.bidirectional = False
        self.batch_first = True
        self.dropout = 0.0
        self.is_train = True

        num_directions = 2 if self.bidirectional else 1
        total_layers = self.num_layers * num_directions
        self.w_ih = [torch.randn(self.hidden_size, self.input_size if layer == 0 else self.hidden_size * num_directions) for layer in range(total_layers)]
        self.w_hh = [torch.randn(self.hidden_size, self.hidden_size) for _ in range(total_layers)]
        self.b_ih = [torch.randn(self.hidden_size) for _ in range(total_layers)]
        self.b_hh = [torch.randn(self.hidden_size) for _ in range(total_layers)]

        self.input = torch.randn(self.batch_size, self.seq_len, self.input_size)

    def test_rnnbase_lstm(self):
        model = rnnbase(mode="LSTM", input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional, is_train=self.is_train, w_ih=self.w_ih, w_hh=self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        output, (h_n, c_n) = model(self.input, None)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h_n.shape, (self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size))
        self.assertEqual(c_n.shape, (self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size))

    def test_rnnbase_gru(self):
        model = rnnbase(mode="GRU", input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional, is_train=self.is_train, w_ih=self.w_ih, w_hh=self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        output, h_n = model(self.input, None)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h_n.shape, (self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size))


class TestLayerNorm(CustomTestCase):

    def test_layernorm(self):
        input_shape = (2, 3, 4)
        normalized_shape = (4,)
        gamma = torch.ones(normalized_shape)
        beta = torch.zeros(normalized_shape)
        eps = 1e-5
        inputs = torch.randn(input_shape)
        ln = layernorm(normalized_shape, gamma, beta, eps, input_shape)
        output = ln(inputs)
        expected = F.layer_norm(inputs, normalized_shape, gamma, beta, eps)
        self.assertTrue(torch.allclose(output, expected))


class TestMultiHeadAttention(CustomTestCase):

    def test_multiheadattention(self):
        embed_dim = 8
        num_heads = 2
        dropout = 0.0
        batch_first = False
        need_weights = False
        q_weight = torch.randn(embed_dim, embed_dim)
        k_weight = torch.randn(embed_dim, embed_dim)
        v_weight = torch.randn(embed_dim, embed_dim)
        out_weight = torch.randn(embed_dim, embed_dim)
        q_bias = torch.randn(embed_dim)
        k_bias = torch.randn(embed_dim)
        v_bias = torch.randn(embed_dim)
        out_bias = torch.randn(embed_dim)
        train = True
        mha = multiheadattention(embed_dim, num_heads, dropout, batch_first, need_weights, q_weight, k_weight, v_weight, out_weight, q_bias, k_bias, v_bias, out_bias, train)
        q = torch.randn(5, 3, embed_dim)
        k = torch.randn(5, 3, embed_dim)
        v = torch.randn(5, 3, embed_dim)
        attn_output, _ = mha(q, k, v, None, None)
        self.assertEqual(attn_output.shape, (5, 3, embed_dim))


class TestActivations(CustomTestCase):

    def test_prelu(self):
        input_tensor = torch.randn(2, 3, 4, 4)
        weight = torch.randn(1)
        output = prelu(input_tensor, weight, data_format="channels_first")
        expected = F.prelu(input_tensor, weight)
        self.assertTrue(torch.allclose(output, expected))

    def test_hardsigmoid(self):
        input_tensor = torch.randn(10)
        output = hardsigmoid(input_tensor)
        expected = F.hardsigmoid(input_tensor)
        self.assertTrue(torch.allclose(output, expected))

    def test_hardswish(self):
        input_tensor = torch.randn(10)
        output = hardswish(input_tensor)
        expected = F.hardswish(input_tensor)
        self.assertTrue(torch.allclose(output, expected))

    def test_swish(self):
        input_tensor = torch.randn(10)
        output = swish(input_tensor)
        expected = input_tensor * torch.sigmoid(input_tensor)
        self.assertTrue(torch.allclose(output, expected))


class TestLinearAndUnfold(CustomTestCase):

    def test_linear(self):
        input_tensor = torch.randn(5, 3)
        weight = torch.randn(4, 3)
        bias = torch.randn(4)
        output = linear(input_tensor, weight, bias)
        expected = F.linear(input_tensor, weight, bias)
        self.assertTrue(torch.allclose(output, expected))

    def test_unfold(self):
        input_tensor = torch.randn(1, 3, 10, 10)
        kernel_size = (3, 3)
        output = unfold(input_tensor, kernel_size)
        expected = F.unfold(input_tensor, kernel_size)
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
