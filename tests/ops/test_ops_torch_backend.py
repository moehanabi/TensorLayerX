import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "torch"

import numpy as np
import torch

from tensorlayerx.backend.ops.torch_backend import *
from tests.utils import CustomTestCase


class TestTorchBackend(CustomTestCase):
    def test_zeros(self):
        shape = (2, 3)
        result = zeros(shape)
        expected = np.zeros(shape)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_ones(self):
        shape = (2, 3)
        result = ones(shape)
        expected = np.ones(shape)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_constant(self):
        shape = (2, 3)
        value = 5
        result = constant(value, shape=shape)
        expected = np.full(shape, value)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_random_uniform(self):
        shape = (2, 3)
        minval = 0
        maxval = 1
        result = random_uniform(shape, minval, maxval)
        self.assertEqual(result.shape, torch.Size(shape))
        self.assertTrue((result >= minval).all() and (result < maxval).all())

    def test_random_normal(self):
        shape = (2, 3)
        mean = 0.0
        stddev = 1.0
        result = random_normal(shape, mean, stddev)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_truncated_normal(self):
        shape = (2, 3)
        mean = 0.0
        stddev = 1.0
        result = truncated_normal(shape, mean, stddev)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_he_normal(self):
        shape = (2, 3)
        result = he_normal(shape)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_he_uniform(self):
        shape = (2, 3)
        result = he_uniform(shape)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_xavier_normal(self):
        shape = (2, 3)
        result = xavier_normal(shape)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_xavier_uniform(self):
        shape = (2, 3)
        result = xavier_uniform(shape)
        self.assertEqual(result.shape, torch.Size(shape))

    def test_matmul(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_add(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = add(a, b)
        expected = np.array([5, 7, 9])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_minimum(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([3, 2, 1])
        result = minimum(a, b)
        expected = np.array([1, 2, 1])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reshape(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        shape = (3, 2)
        result = reshape(tensor, shape)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_concat(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        result = concat([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_convert_to_tensor(self):
        value = [1, 2, 3]
        result = convert_to_tensor(value)
        expected = np.array(value)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_sqrt(self):
        x = torch.tensor([1.0, 4.0, 9.0])
        result = sqrt(x)
        expected = np.sqrt([1.0, 4.0, 9.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_mean(x)
        expected = np.mean([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_max(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_max(x)
        expected = np.max([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_min(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_min(x)
        expected = np.min([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    # def test_pad(self):
    #     x = torch.tensor([[1, 2], [3, 4]])
    #     paddings = ((1, 1), (2, 2))
    #     result = pad(x, paddings, mode="CONSTANT", constant_values=0)
    #     expected = np.pad(x.numpy(), paddings, mode="constant", constant_values=0)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_stack(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([3, 4])
        result = stack([a, b], axis=0)
        expected = np.stack([a.numpy(), b.numpy()], axis=0)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    # def test_meshgrid(self):
    #     a = torch.tensor([1, 2, 3])
    #     b = torch.tensor([4, 5, 6])
    #     result = meshgrid(a, b)
    #     expected = np.meshgrid(a.numpy(), b.numpy())
    #     self.assertTrue(all(np.array_equal(r.numpy(), e) for r, e in zip(result, expected)))

    def test_arange(self):
        start = 0
        limit = 5
        delta = 1
        result = arange(start, limit, delta)
        expected = np.arange(start, limit, delta)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_expand_dims(self):
        x = torch.tensor([1, 2, 3])
        axis = 0
        result = expand_dims(x, axis)
        expected = np.expand_dims(x.numpy(), axis)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_tile(self):
        x = torch.tensor([1, 2, 3])
        multiples = (2, 2)
        result = tile(x, multiples)
        expected = np.tile(x.numpy(), multiples)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_cast(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        dtype = torch.int32
        result = cast(x, dtype)
        expected = x.numpy().astype(np.int32)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_transpose(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        perm = (1, 0)
        result = transpose(x, perm)
        expected = np.transpose(x.numpy(), perm)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_gather_nd(self):
        params = torch.tensor([[1, 2], [3, 4]])
        indices = torch.tensor([[0, 0], [1, 1]])
        result = gather_nd(params, indices)
        expected = np.array([1, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_clip_by_value(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        clip_value_min = 1.5
        clip_value_max = 2.5
        result = clip_by_value(t, clip_value_min, clip_value_max)
        expected = np.clip(t.numpy(), clip_value_min, clip_value_max)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_split(self):
        inputs = torch.randn(10, 20)
        size_splits = [2, 3, 5]
        outputs = split(inputs, size_splits, axis=0)
        self.assertEqual(len(outputs), len(size_splits), f"Expected {len(size_splits)} splits, but got {len(outputs)}")
        for out, size in zip(outputs, size_splits):
            self.assertEqual(out.shape, (size, 20), f"Expected shape ({size}, 20), but got {out.shape}")

    def test_floor(self):
        x = torch.tensor([1.7, 2.3, 3.9])
        result = floor(x)
        expected = np.floor(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_gather(self):
        params = torch.tensor([1, 2, 3, 4])
        indices = torch.tensor([0, 2])
        result = gather(params, indices)
        expected = params.numpy()[indices.numpy()]
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_linspace(self):
        start = 0
        stop = 5
        num = 6
        result = linspace(start, stop, num)
        expected = np.linspace(start, stop, num)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_slice(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        starts = [0, 1]
        sizes = [2, 1]
        result = slice(x, starts, sizes)
        expected = x[0:2, 1:2]
        self.assertTrue(torch.equal(result, expected))

    def test_add_n(self):
        inputs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = add_n(inputs)
        expected = torch.tensor([4, 6])
        self.assertTrue(torch.equal(result, expected))

    def test_one_hot(self):
        x = torch.tensor([0, 1, 2])
        depth = 3
        result = OneHot(depth)(x)
        expected = np.eye(depth)[x.numpy()]
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_l2_normalize(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        axis = 0
        result = l2_normalize(x, axis)
        expected = x.numpy() / np.linalg.norm(x.numpy(), ord=2, axis=axis, keepdims=True)
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_not_equal(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 2, 4])
        result = NotEqual()(x, y)
        expected = np.not_equal(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_count_nonzero(self):
        x = torch.tensor([0, 1, 2, 0, 3])
        result = CountNonzero()(x)
        expected = np.count_nonzero(x.numpy())
        self.assertEqual(result, expected)

    def test_resize(self):
        inputs = torch.randn(1, 3, 24, 24)
        output_size = (48, 48)
        method = "bilinear"
        antialias = True
        output = resize(inputs, output_size, method, antialias)
        self.assertEqual(output.shape, (1, 3, 48, 48))

    # def test_zero_padding_1d(self):
    #     x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    #     padding = (1, 1)
    #     data_format = "channels_last"
    #     result = ZeroPadding1D(padding, data_format)(x)
    #     expected = np.pad(x.numpy(), ((0, 0), (1, 1)), mode="constant", constant_values=0)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    # def test_zero_padding_2d(self):
    #     x = torch.tensor([[1, 2], [3, 4]])
    #     padding = ((1, 1), (2, 2))
    #     data_format = "channels_last"
    #     result = ZeroPadding2D(padding, data_format)(x)
    #     expected = np.pad(x.numpy(), ((1, 1), (2, 2)), mode="constant", constant_values=0)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    # def test_zero_padding_3d(self):
    #     x = torch.randn(1, 1, 2, 2, 2)
    #     padding = ((1, 1), (1, 1), (1, 1))
    #     data_format = "channels_last"
    #     result = ZeroPadding3D(padding, data_format)(x)
    #     expected = np.pad(x.numpy(), ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_sign(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = Sign()(x)
        expected = np.sign(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_ceil(self):
        x = torch.tensor([1.2, 2.5, 3.7])
        result = ceil(x)
        expected = np.ceil(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_multiply(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        result = multiply(x, y)
        expected = np.multiply(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_divide(self):
        x = torch.tensor([4, 5, 6])
        y = torch.tensor([2, 2, 2])
        result = divide(x, y)
        expected = np.divide(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_triu(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = triu(x)
        expected = np.triu(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_tril(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = tril(x)
        expected = np.tril(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_abs(self):
        x = torch.tensor([-1.0, -2.0, 3.0])
        result = abs(x)
        expected = np.abs(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_acos(self):
        x = torch.tensor([1.0, 0.0, -1.0])
        result = acos(x)
        expected = np.arccos(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_acosh(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = acosh(x)
        expected = np.arccosh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_angle(self):
        x = torch.tensor([1.0 + 1.0j, 1.0 - 1.0j])
        result = angle(x)
        expected = np.angle(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_argmax(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = argmax(x)
        expected = np.argmax(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_argmin(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = argmin(x)
        expected = np.argmin(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_asin(self):
        x = torch.tensor([1.0, 0.0, -1.0])
        result = asin(x)
        expected = np.arcsin(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_asinh(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = asinh(x)
        expected = np.arcsinh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_atan(self):
        x = torch.tensor([1.0, 0.0, -1.0])
        result = atan(x)
        expected = np.arctan(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_atanh(self):
        x = torch.tensor([0.5, 0.0, -0.5])
        result = atanh(x)
        expected = np.arctanh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_cos(self):
        x = torch.tensor([0.0, np.pi / 2, np.pi])
        result = cos(x)
        expected = np.cos(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_cosh(self):
        x = torch.tensor([0.0, 1.0, -1.0])
        result = cosh(x)
        expected = np.cosh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_count_nonzero(self):
        x = torch.tensor([0, 1, 2, 0, 3])
        result = count_nonzero(x)
        expected = np.count_nonzero(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_cumprod(self):
        x = torch.tensor([1, 2, 3, 4])
        result = cumprod(x)
        expected = np.cumprod(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_cumsum(self):
        x = torch.tensor([1, 2, 3, 4])
        result = cumsum(x)
        expected = np.cumsum(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_equal(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 2, 3])
        self.assertTrue(equal(x, y))

    def test_exp(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = exp(x)
        expected = np.exp(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_floordiv(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 2, 2])
        result = floordiv(x, y)
        expected = np.floor_divide(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_floormod(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 2, 2])
        result = floormod(x, y)
        expected = np.mod(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_greater(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 8, 6])
        result = greater(x, y)
        expected = np.greater(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_greater_equal(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 8, 9])
        result = greater_equal(x, y)
        expected = np.greater_equal(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_is_inf(self):
        x = torch.tensor([1.0, float("inf"), 2.0])
        result = is_inf(x)
        expected = np.isinf(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_is_nan(self):
        x = torch.tensor([1.0, float("nan"), 2.0])
        result = is_nan(x)
        expected = np.isnan(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_l2_normalize(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = l2_normalize(x)
        expected = x / np.linalg.norm(x.numpy(), ord=2)
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_less(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 8, 6])
        result = less(x, y)
        expected = np.less(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_less_equal(self):
        x = torch.tensor([5, 7, 9])
        y = torch.tensor([2, 8, 9])
        result = less_equal(x, y)
        expected = np.less_equal(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_log(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = log(x)
        expected = np.log(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_log_sigmoid(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-x.numpy())))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_maximum(self):
        x = torch.tensor([1, 4, 3])
        y = torch.tensor([2, 2, 5])
        result = maximum(x, y)
        expected = np.maximum(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_negative(self):
        x = torch.tensor([1, -2, 3])
        result = negative(x)
        expected = -x.numpy()
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_not_equal(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 2, 4])
        result = not_equal(x, y)
        expected = np.not_equal(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_pow(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([2, 2, 2])
        result = pow(x, y)
        expected = np.power(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_real(self):
        x = torch.tensor([1 + 2j, 3 + 4j])
        result = real(x)
        expected = np.real(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reciprocal(self):
        x = torch.tensor([1.0, 2.0, 4.0])
        result = reciprocal(x)
        expected = 1 / x.numpy()
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_prod(self):
        x = torch.tensor([1, 2, 3, 4])
        result = reduce_prod(x)
        expected = np.prod(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_reduce_std(self):
        tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reduce_std(tensor_1d)
        expected = torch.std(tensor_1d)
        self.assertTrue(torch.allclose(result, expected))

    def test_reduce_sum(self):
        tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reduce_sum(tensor_1d)
        expected = torch.sum(tensor_1d)
        self.assertTrue(torch.allclose(result, expected))

    def test_reduce_variance(self):
        tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reduce_variance(tensor_1d)
        expected = torch.var(tensor_1d)
        self.assertTrue(torch.allclose(result, expected))

    def test_round(self):
        x = torch.tensor([1.2, 2.5, 3.7])
        result = round(x)
        expected = np.round(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_rsqrt(self):
        x = torch.tensor([1.0, 4.0, 9.0])
        result = rsqrt(x)
        expected = 1 / np.sqrt(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_segment_max(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        result = segment_max(x, segment_ids)
        expected = torch.tensor([2, 5])
        self.assertTrue(torch.equal(result, expected))

    def test_segment_mean(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        result = segment_mean(x, segment_ids)
        expected = torch.tensor([1.5, 4.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_segment_min(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        result = segment_min(x, segment_ids)
        expected = torch.tensor([1, 3])
        self.assertTrue(torch.equal(result, expected))

    def test_segment_sum(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        result = segment_sum(x, segment_ids)
        expected = torch.tensor([3, 12])
        self.assertTrue(torch.equal(result, expected))

    def test_sigmoid(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-x.numpy()))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_sign(self):
        x = torch.tensor([1.0, -2.0, 0.0])
        result = sign(x)
        expected = np.sign(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_sin(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sin(x)
        expected = np.sin(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_sinh(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sinh(x)
        expected = np.sinh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_softplus(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = softplus(x)
        expected = np.log(1 + np.exp(x.numpy()))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_square(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = square(x)
        expected = np.square(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_squared_difference(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 4.0])
        result = squared_difference(x, y)
        expected = np.square(x.numpy() - y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_subtract(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 4.0])
        result = subtract(x, y)
        expected = x.numpy() - y.numpy()
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_tan(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = tan(x)
        expected = np.tan(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_tanh(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = tanh(x)
        expected = np.tanh(x.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_any(self):
        x = torch.tensor([True, False, True])
        result = any(x)
        expected = np.any(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_all(self):
        x = torch.tensor([True, False, True])
        result = all(x)
        expected = np.all(x.numpy())
        self.assertEqual(result.item(), expected)

    def test_logical_and(self):
        x = torch.tensor([True, False, True])
        y = torch.tensor([True, True, False])
        result = logical_and(x, y)
        expected = np.logical_and(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_or(self):
        x = torch.tensor([True, False, True])
        y = torch.tensor([True, True, False])
        result = logical_or(x, y)
        expected = np.logical_or(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_not(self):
        x = torch.tensor([True, False, True])
        result = logical_not(x)
        expected = np.logical_not(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_xor(self):
        x = torch.tensor([True, False, True])
        y = torch.tensor([True, True, False])
        result = logical_xor(x, y)
        expected = np.logical_xor(x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_argsort(self):
        x = torch.tensor([3, 1, 2])
        result = argsort(x)
        expected = np.argsort(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_bmm(self):
        x = torch.randn(10, 3, 4)
        y = torch.randn(10, 4, 5)
        result = bmm(x, y)
        expected = np.matmul(x.numpy(), y.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_where(self):
        condition = torch.tensor([True, False, True])
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        result = where(condition, x, y)
        expected = np.array([1, 5, 3])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_ones_like(self):
        x = torch.tensor([1, 2, 3])
        result = ones_like(x)
        expected = np.ones_like(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_zeros_like(self):
        x = torch.tensor([1, 2, 3])
        result = zeros_like(x)
        expected = np.zeros_like(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_squeeze(self):
        tensor = torch.tensor([[[1], [2]], [[3], [4]]])
        result = squeeze(tensor, axis=2)
        expected = torch.tensor([[1, 2], [3, 4]])
        self.assertTrue(torch.equal(result, expected))

    def test_unsorted_segment_sum(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        num_segments = 2
        result = unsorted_segment_sum(x, segment_ids, num_segments)
        expected = torch.tensor([3, 12])
        self.assertTrue(torch.equal(result, expected))

    def test_unsorted_segment_mean(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        num_segments = 2
        result = unsorted_segment_mean(x, segment_ids, num_segments)
        expected = torch.tensor([1.5, 4.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_unsorted_segment_min(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        num_segments = 2
        result = unsorted_segment_min(x, segment_ids, num_segments)
        expected = torch.tensor([1, 3])
        self.assertTrue(torch.equal(result, expected))

    def test_unsorted_segment_max(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        segment_ids = [0, 0, 1, 1, 1]
        num_segments = 2
        result = unsorted_segment_max(x, segment_ids, num_segments)
        expected = torch.tensor([2, 5])
        self.assertTrue(torch.equal(result, expected))

    def test_set_seed(self):
        set_seed(42)
        x = torch.randn(1)
        set_seed(42)
        y = torch.randn(1)
        self.assertTrue(torch.equal(x, y))

    def test_is_tensor(self):
        x = torch.tensor([1, 2, 3])
        self.assertTrue(is_tensor(x))

    def test_tensor_scatter_nd_update(self):
        tensor = torch.tensor([1, 2, 3, 4, 5])
        indices = torch.tensor([0, 2])
        updates = torch.tensor([9, 10])
        result = tensor_scatter_nd_update(tensor, indices, updates)
        expected = torch.tensor([9, 2, 10, 4, 5])
        self.assertTrue(torch.equal(result, expected))

    def test_diag(self):
        x = torch.tensor([1, 2, 3])
        result = diag(x)
        expected = np.diag(x.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_mask_select(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        mask = torch.tensor([True, False, True, False, True])
        result = mask_select(x, mask)
        expected = np.array([1, 3, 5])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_eye(self):
        result = eye(3)
        expected = np.eye(3)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_einsum(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[1, 2], [3, 4]])
        result = einsum("ij,jk->ik", x, y)
        expected = np.einsum("ij,jk->ik", x.numpy(), y.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    # def test_scatter_update(self):
    #     tensor = torch.tensor([1, 2, 3, 4, 5])
    #     indices = torch.tensor([0, 2])
    #     updates = torch.tensor([9, 10])
    #     result = scatter_update(tensor, indices, updates)
    #     expected = torch.tensor([9, 2, 10, 4, 5])
    #     self.assertTrue(torch.equal(result, expected))

    # def test_roll(self):
    #     x = torch.tensor([1, 2, 3, 4, 5])
    #     result = roll(x, shifts=2)
    #     expected = np.roll(x.numpy(), shifts=2)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logsoftmax(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = logsoftmax(x)
        expected = torch.nn.functional.log_softmax(x).numpy()
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_topk(self):
        input_tensor = torch.tensor([1, 3, 5, 7, 9])
        values, indices = topk(input_tensor, 3)
        self.assertTrue(torch.equal(values, torch.tensor([9, 7, 5])))
        self.assertTrue(torch.equal(indices, torch.tensor([4, 3, 2])))

    def test_numel(self):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(numel(input_tensor), 6)

    def test_expand(self):
        input_tensor = torch.tensor([1, 2, 3])
        expanded_tensor = expand(input_tensor, (3, 3))
        expected_tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.assertTrue(torch.equal(expanded_tensor, expected_tensor))

    def test_flip(self):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        flipped_tensor = flip(input_tensor, [0])
        expected_tensor = torch.tensor([[4, 5, 6], [1, 2, 3]])
        self.assertTrue(torch.equal(flipped_tensor, expected_tensor))

    def test_mv(self):
        matrix = torch.tensor([[1, 2], [3, 4]])
        vector = torch.tensor([1, 2])
        result = mv(matrix, vector)
        expected_result = torch.tensor([5, 11])
        self.assertTrue(torch.equal(result, expected_result))


if __name__ == "__main__":
    unittest.main()
