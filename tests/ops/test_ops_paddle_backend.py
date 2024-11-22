import os
import paddle
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "padddle"
import numpy as np

from tensorlayerx.backend.ops.paddle_backend import *
from tests.utils import CustomTestCase


class TestPaddleBackend(CustomTestCase):

    def test_zeros(self):
        shape = [2, 3]
        result = zeros(shape)
        expected = np.zeros(shape)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones(self):
        shape = [2, 3]
        result = ones(shape)
        expected = np.ones(shape)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_constant(self):
        value = 5
        shape = [2, 3]
        result = constant(value, shape=shape)
        expected = np.full(shape, value)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_random_uniform(self):
        shape = [2, 3]
        minval = 0
        maxval = 1
        result = random_uniform(shape, minval=minval, maxval=maxval)
        self.assertEqual(result.shape, shape)
        self.assertTrue(np.all(result.numpy() >= minval))
        self.assertTrue(np.all(result.numpy() < maxval))

    def test_random_normal(self):
        shape = [2, 3]
        mean = 0
        stddev = 1
        result = random_normal(shape, mean=mean, stddev=stddev)
        self.assertEqual(result.shape, shape)
        self.assertAlmostEqual(np.mean(result.numpy()), mean, delta=0.1)
        self.assertAlmostEqual(np.std(result.numpy()), stddev, delta=0.1)

    # def test_get_tensor_shape(self):
    #     tensor = paddle.to_tensor(np.zeros([2, 3]))
    #     result = get_tensor_shape(tensor)
    #     expected = [2, 3]
    #     self.assertEqual(result, expected)

    def test_matmul(self):
        a = paddle.to_tensor(np.array([[1, 2], [3, 4]]))
        b = paddle.to_tensor(np.array([[5, 6], [7, 8]]))
        result = matmul(a, b)
        expected = np.matmul(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_add(self):
        a = paddle.to_tensor(np.array([1, 2, 3]))
        b = paddle.to_tensor(np.array([4, 5, 6]))
        result = add(a, b)
        expected = np.add(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_minimum(self):
        a = paddle.to_tensor(np.array([1, 2, 3]))
        b = paddle.to_tensor(np.array([3, 2, 1]))
        result = minimum(a, b)
        expected = np.minimum(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reshape(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4, 5, 6]))
        shape = [2, 3]
        result = reshape(tensor, shape)
        expected = np.reshape(tensor.numpy(), shape)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_concat(self):
        a = paddle.to_tensor(np.array([[1, 2], [3, 4]]))
        b = paddle.to_tensor(np.array([[5, 6], [7, 8]]))
        result = concat([a, b], axis=0)
        expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_convert_to_tensor(self):
        value = np.array([1, 2, 3])
        result = convert_to_tensor(value)
        np.testing.assert_array_equal(result.numpy(), value)

    def test_convert_to_numpy(self):
        value = paddle.to_tensor(np.array([1, 2, 3]))
        result = convert_to_numpy(value)
        expected = value.numpy()
        np.testing.assert_array_equal(result, expected)

    def test_sqrt(self):
        value = paddle.to_tensor(np.array([1, 4, 9]))
        result = sqrt(value)
        expected = np.sqrt(value.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_mean(self):
        value = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result = reduce_mean(value, axis=0)
        expected = np.mean(value.numpy(), axis=0)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_max(self):
        value = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result = reduce_max(value, axis=0)
        expected = np.max(value.numpy(), axis=0)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_min(self):
        value = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result = reduce_min(value, axis=0)
        expected = np.min(value.numpy(), axis=0)
        np.testing.assert_array_equal(result.numpy(), expected)

    # def test_pad(self):
    #     tensor = paddle.to_tensor(np.array([[1, 2], [3, 4]]))
    #     paddings = [[1, 1], [2, 2]]
    #     result = pad(tensor, paddings, mode="CONSTANT", constant_values=0)
    #     expected = np.pad(tensor.numpy(), paddings, mode="constant", constant_values=0)
    #     np.testing.assert_array_equal(result.numpy(), expected)

    def test_stack(self):
        a = paddle.to_tensor(np.array([1, 2]))
        b = paddle.to_tensor(np.array([3, 4]))
        result = stack([a, b], axis=0)
        expected = np.stack([a.numpy(), b.numpy()], axis=0)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_arange(self):
        start = 0
        limit = 5
        delta = 1
        result = arange(start, limit, delta)
        expected = np.arange(start, limit, delta)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_expand_dims(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3]))
        axis = 0
        result = expand_dims(tensor, axis)
        expected = np.expand_dims(tensor.numpy(), axis)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_tile(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3]))
        multiples = [2]
        result = tile(tensor, multiples)
        expected = np.tile(tensor.numpy(), multiples)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cast(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3]), dtype="int32")
        dtype = "float32"
        result = cast(tensor, dtype)
        expected = tensor.numpy().astype(dtype)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_transpose(self):
        tensor = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        perm = [1, 0]
        result = transpose(tensor, perm)
        expected = np.transpose(tensor.numpy(), perm)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_gather_nd(self):
        params = paddle.to_tensor(np.array([[1, 2], [3, 4]]))
        indices = paddle.to_tensor(np.array([[0, 1], [1, 0]]))
        result = gather_nd(params, indices)
        expected = np.array([2, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_clip_by_value(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4, 5]))
        clip_value_min = 2
        clip_value_max = 4
        result = clip_by_value(tensor, clip_value_min, clip_value_max)
        expected = np.clip(tensor.numpy(), clip_value_min, clip_value_max)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_split(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4, 5, 6]))
        num_or_size_splits = 3
        result = split(tensor, num_or_size_splits)
        expected = np.split(tensor.numpy(), num_or_size_splits)
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r.numpy(), e)

    def test_floor(self):
        tensor = paddle.to_tensor(np.array([1.2, 2.5, 3.7]))
        result = floor(tensor)
        expected = np.floor(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_gather(self):
        params = paddle.to_tensor(np.array([1, 2, 3, 4, 5]))
        indices = paddle.to_tensor(np.array([0, 2, 4]))
        result = gather(params, indices)
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_linspace(self):
        start = 0
        stop = 10
        num = 5
        result = linspace(start, stop, num)
        expected = np.linspace(start, stop, num)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_slice(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4, 5]))
        axes = [0]
        starts = [1]
        sizes = [3]
        result = slice(tensor, axes, starts, sizes)
        expected = tensor.numpy()[1:4]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_add_n(self):
        tensors = [paddle.to_tensor(np.array([1, 2, 3])), paddle.to_tensor(np.array([4, 5, 6]))]
        result = add_n(tensors)
        expected = np.add(tensors[0].numpy(), tensors[1].numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_one_hot(self):
        indices = paddle.to_tensor(np.array([0, 1, 2]))
        depth = 3
        result = OneHot(depth=depth)(indices)
        expected = np.eye(depth)[indices.numpy()]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_l2_normalize(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3]))
        result = L2Normalize(axis=0)(tensor)
        expected = tensor.numpy() / np.linalg.norm(tensor.numpy(), ord=2, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_not_equal(self):
        a = paddle.to_tensor(np.array([1, 2, 3]))
        b = paddle.to_tensor(np.array([3, 2, 1]))
        result = NotEqual()(a, b)
        expected = np.not_equal(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_count_nonzero(self):
        input_tensor = pd.to_tensor([[0, 1, 2], [3, 0, 4]], dtype="int64")
        count_nonzero = CountNonzero()
        result = count_nonzero(input_tensor)
        expected = (pd.to_tensor([0, 0, 1, 1, 1]), pd.to_tensor([1, 2, 0, 2, 2]))
        self.assertTrue(np.array_equal(result[0].numpy(), expected[0].numpy()))
        self.assertTrue(np.array_equal(result[1].numpy(), expected[1].numpy()))

    def test_resize(self):
        tensor = paddle.to_tensor(np.random.rand(1, 3, 4, 4))
        scale = [2, 2]
        result = Resize(scale=scale, method="nearest")(tensor)
        expected = np.kron(tensor.numpy(), np.ones((1, 1, 2, 2)))
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    # def test_zero_padding_1d(self):
    #     tensor = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    #     padding = [1, 1]
    #     result = ZeroPadding1D(padding, data_format="channels_last")(tensor)
    #     expected = np.pad(tensor.numpy(), ((0, 0), (1, 1)), mode="constant")
    #     np.testing.assert_array_equal(result.numpy(), expected)

    # def test_zero_padding_2d(self):
    #     tensor = paddle.to_tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    #     padding = [[1, 1], [1, 1]]
    #     result = ZeroPadding2D(padding, data_format="channels_last")(tensor)
    #     expected = np.pad(tensor.numpy(), ((0, 0), (1, 1), (1, 1)), mode="constant")
    #     np.testing.assert_array_equal(result.numpy(), expected)

    # def test_zero_padding_3d(self):
    #     tensor = paddle.to_tensor(np.random.rand(1, 2, 2, 2))
    #     padding = [[1, 1], [1, 1], [1, 1]]
    #     result = ZeroPadding3D(padding, data_format="channels_last")(tensor)
    #     expected = np.pad(tensor.numpy(), ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant")
    #     np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_ceil(self):
        tensor = paddle.to_tensor(np.array([1.2, 2.5, 3.7]))
        result = ceil(tensor)
        expected = np.ceil(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply(self):
        a = paddle.to_tensor(np.array([1, 2, 3]))
        b = paddle.to_tensor(np.array([4, 5, 6]))
        result = multiply(a, b)
        expected = np.multiply(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_divide(self):
        a = paddle.to_tensor(np.array([4, 5, 6]))
        b = paddle.to_tensor(np.array([1, 2, 3]))
        result = divide(a, b)
        expected = np.divide(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    # def test_identity(self):
    #     tensor = paddle.to_tensor(np.array([1, 2, 3]))
    #     result = identity(tensor)
    #     expected = tensor.numpy()
    #     np.testing.assert_array_equal(result.numpy(), expected)

    def test_triu(self):
        tensor = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = triu(tensor)
        expected = np.triu(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_tril(self):
        tensor = paddle.to_tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = tril(tensor)
        expected = np.tril(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_abs(self):
        tensor = paddle.to_tensor(np.array([-1, -2, -3]))
        result = abs(tensor)
        expected = np.abs(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_acos(self):
        tensor = paddle.to_tensor(np.array([1, 0, -1]))
        result = acos(tensor)
        expected = np.arccos(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_acosh(self):
        x = pd.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        expected = np.log([1.0 + np.sqrt(1.0**2 - 1), 2.0 + np.sqrt(2.0**2 - 1), 3.0 + np.sqrt(3.0**2 - 1), 4.0 + np.sqrt(4.0**2 - 1)])
        result = acosh(x)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_angle(self):
        x = pd.to_tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype="complex64")
        expected = np.angle([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        result = angle(x)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_argmax(self):
        tensor = paddle.to_tensor(np.array([1, 3, 2]))
        result = argmax(tensor)
        expected = np.argmax(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_argmin(self):
        tensor = paddle.to_tensor(np.array([1, 3, 2]))
        result = argmin(tensor)
        expected = np.argmin(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_asin(self):
        x = pd.to_tensor([0.0, 0.5, 1.0])
        expected = np.arcsin([0.0, 0.5, 1.0])
        result = asin(x).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_asinh(self):
        x = pd.to_tensor([0.0, 0.5, 1.0])
        expected = np.arcsinh([0.0, 0.5, 1.0])
        result = asinh(x).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_atan(self):
        x = pd.to_tensor([0.0, 0.5, 1.0])
        expected = np.arctan([0.0, 0.5, 1.0])
        result = atan(x).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_atanh(self):
        x = pd.to_tensor([0.0, 0.5, 0.9])
        expected = np.arctanh([0.0, 0.5, 0.9])
        result = atanh(x).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cos(self):
        tensor = paddle.to_tensor(np.array([0, np.pi / 2, np.pi]))
        result = cos(tensor)
        expected = np.cos(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cosh(self):
        tensor = paddle.to_tensor(np.array([0, np.pi / 2, np.pi]))
        result = cosh(tensor)
        expected = np.cosh(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_count_nonzero(self):
        tensor = paddle.to_tensor(np.array([0, 1, 2, 0, 3]))
        result = count_nonzero(tensor)
        expected = np.count_nonzero(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cumprod(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4]))
        result = cumprod(tensor)
        expected = np.cumprod(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cumsum(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4]))
        result = cumsum(tensor)
        expected = np.cumsum(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal(self):
        a = paddle.to_tensor(np.array([1, 2, 3]))
        b = paddle.to_tensor(np.array([1, 2, 4]))
        result = equal(a, b)
        expected = np.equal(a.numpy(), b.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_exp(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3]))
        result = exp(tensor)
        expected = np.exp(tensor.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_floordiv(self):
        x = paddle.to_tensor([5, 7, 9])
        y = paddle.to_tensor([2, 2, 2])
        result = floordiv(x, y)
        expected = np.array([2, 3, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_floormod(self):
        x = paddle.to_tensor([5, 7, 9])
        y = paddle.to_tensor([2, 2, 2])
        result = floormod(x, y)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 2])
        result = greater(x, y)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 2])
        result = greater_equal(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_is_inf(self):
        x = paddle.to_tensor([1, float("inf"), 3])
        result = is_inf(x)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_is_nan(self):
        x = paddle.to_tensor([1, float("nan"), 3])
        result = is_nan(x)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_l2_normalize(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = l2_normalize(x)
        expected = x.numpy() / np.sqrt(np.sum(x.numpy() ** 2))
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_less(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 2])
        result = less(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 2])
        result = less_equal(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_log(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = log(x)
        expected = np.log([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_log_sigmoid(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-np.array([1, 2, 3]))))
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_maximum(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 1, 4])
        result = maximum(x, y)
        expected = np.maximum([1, 2, 3], [2, 1, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_negative(self):
        x = paddle.to_tensor([1, 2, 3])
        result = negative(x)
        expected = -np.array([1, 2, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 2, 4])
        result = not_equal(x, y)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_pow(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        y = paddle.to_tensor([2, 2, 2], dtype="float32")
        result = pow(x, y)
        expected = np.power([1, 2, 3], [2, 2, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_real(self):
        x = paddle.to_tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype="complex64")
        result = real(x)
        expected = np.real([1 + 2j, 3 + 4j, 5 + 6j])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_reciprocal(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = reciprocal(x)
        expected = 1 / np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_reduce_prod(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = reduce_prod(x)
        expected = np.prod([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_reduce_std(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = reduce_std(x)
        expected = np.std([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_reduce_sum(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = reduce_sum(x)
        expected = np.sum([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_reduce_variance(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = reduce_variance(x)
        expected = np.var([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_round(self):
        x = paddle.to_tensor([1.1, 2.5, 3.7], dtype="float32")
        result = round(x)
        expected = np.round([1.1, 2.5, 3.7])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_rsqrt(self):
        x = paddle.to_tensor([1, 4, 9], dtype="float32")
        result = rsqrt(x)
        expected = 1 / np.sqrt([1, 4, 9])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_segment_max(self):
        x = pd.to_tensor([1, 2, 3, 4, 5], dtype="float32")
        segment_ids = pd.to_tensor([0, 0, 1, 1, 1], dtype="int32")
        expected = np.array([2, 5], dtype="float32")
        result = segment_max(x, segment_ids).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_segment_mean(self):
        x = pd.to_tensor([1, 2, 3, 4, 5], dtype="float32")
        segment_ids = pd.to_tensor([0, 0, 1, 1, 1], dtype="int32")
        expected = np.array([1.5, 4], dtype="float32")
        result = segment_mean(x, segment_ids).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_segment_min(self):
        x = pd.to_tensor([1, 2, 3, 4, 5], dtype="float32")
        segment_ids = pd.to_tensor([0, 0, 1, 1, 1], dtype="int32")
        expected = np.array([1, 3], dtype="float32")
        result = segment_min(x, segment_ids).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_segment_sum(self):
        x = pd.to_tensor([1, 2, 3, 4, 5], dtype="float32")
        segment_ids = pd.to_tensor([0, 0, 1, 1, 1], dtype="int32")
        expected = np.array([3, 12], dtype="float32")
        result = segment_sum(x, segment_ids).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sigmoid(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-np.array([1, 2, 3])))
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_sign(self):
        x = paddle.to_tensor([-1, 0, 1], dtype="float32")
        result = sign(x)
        expected = np.sign([-1, 0, 1])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_sin(self):
        x = paddle.to_tensor([0, np.pi / 2, np.pi], dtype="float32")
        result = sin(x)
        expected = np.sin([0, np.pi / 2, np.pi])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_sinh(self):
        x = paddle.to_tensor([0, 1, 2], dtype="float32")
        result = sinh(x)
        expected = np.sinh([0, 1, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_softplus(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = softplus(x)
        expected = np.log(1 + np.exp([1, 2, 3]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_square(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = square(x)
        expected = np.square([1, 2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_squared_difference(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        y = paddle.to_tensor([2, 2, 2], dtype="float32")
        result = squared_difference(x, y)
        expected = np.square([1, 2, 3] - [2, 2, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_subtract(self):
        x = paddle.to_tensor([1, 2, 3], dtype="float32")
        y = paddle.to_tensor([2, 2, 2], dtype="float32")
        result = subtract(x, y)
        expected = np.subtract([1, 2, 3], [2, 2, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_tan(self):
        x = paddle.to_tensor([0, np.pi / 4, np.pi / 2], dtype="float32")
        result = tan(x)
        expected = np.tan([0, np.pi / 4, np.pi / 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_tanh(self):
        x = paddle.to_tensor([0, 1, 2], dtype="float32")
        result = tanh(x)
        expected = np.tanh([0, 1, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_any(self):
        x = paddle.to_tensor([False, True, False])
        result = any(x)
        expected = np.any([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_all(self):
        x = paddle.to_tensor([True, True, False])
        result = all(x)
        expected = np.all([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_and(self):
        x = paddle.to_tensor([True, False, True])
        y = paddle.to_tensor([True, True, False])
        result = logical_and(x, y)
        expected = np.logical_and([True, False, True], [True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_or(self):
        x = paddle.to_tensor([True, False, True])
        y = paddle.to_tensor([True, True, False])
        result = logical_or(x, y)
        expected = np.logical_or([True, False, True], [True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_not(self):
        x = paddle.to_tensor([True, False, True])
        result = logical_not(x)
        expected = np.logical_not([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_xor(self):
        x = paddle.to_tensor([True, False, True])
        y = paddle.to_tensor([True, True, False])
        result = logical_xor(x, y)
        expected = np.logical_xor([True, False, True], [True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_argsort(self):
        x = paddle.to_tensor([3, 1, 2])
        result = argsort(x)
        expected = np.argsort([3, 1, 2])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_bmm(self):
        x = paddle.to_tensor(np.random.rand(2, 3, 4), dtype="float32")
        y = paddle.to_tensor(np.random.rand(2, 4, 5), dtype="float32")
        result = bmm(x, y)
        expected = np.matmul(x.numpy(), y.numpy())
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_where(self):
        condition = paddle.to_tensor([True, False, True])
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])
        result = where(condition, x, y)
        expected = np.where([True, False, True], [1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones_like(self):
        x = paddle.to_tensor([1, 2, 3])
        result = ones_like(x)
        expected = np.ones_like([1, 2, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_zeros_like(self):
        x = paddle.to_tensor([1, 2, 3])
        result = zeros_like(x)
        expected = np.zeros_like([1, 2, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_squeeze(self):
        x = paddle.to_tensor([[[1], [2], [3]]])
        result = squeeze(x)
        expected = np.squeeze([[[1], [2], [3]]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unsorted_segment_sum(self):
        x = paddle.to_tensor([1, 2, 3, 4], dtype="float32")
        segment_ids = paddle.to_tensor([0, 1, 0, 1], dtype="int64")
        num_segments = 2
        result = unsorted_segment_sum(x, segment_ids, num_segments)
        expected = np.array([4, 6])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_unsorted_segment_mean(self):
        x = paddle.to_tensor([1, 2, 3, 4], dtype="float32")
        segment_ids = paddle.to_tensor([0, 1, 0, 1], dtype="int64")
        num_segments = 2
        result = unsorted_segment_mean(x, segment_ids, num_segments)
        expected = np.array([2, 3])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_unsorted_segment_min(self):
        x = paddle.to_tensor([1, 2, 3, 4], dtype="float32")
        segment_ids = paddle.to_tensor([0, 1, 0, 1], dtype="int64")
        num_segments = 2
        result = unsorted_segment_min(x, segment_ids, num_segments)
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_unsorted_segment_max(self):
        x = paddle.to_tensor([1, 2, 3, 4], dtype="float32")
        segment_ids = paddle.to_tensor([0, 1, 0, 1], dtype="int64")
        num_segments = 2
        result = unsorted_segment_max(x, segment_ids, num_segments)
        expected = np.array([3, 4])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_set_seed(self):
        set_seed(42)
        result = np.random.rand()
        expected = np.random.RandomState(42).rand()
        self.assertAlmostEqual(result, expected)

    def test_is_tensor(self):
        x = paddle.to_tensor([1, 2, 3])
        self.assertTrue(is_tensor(x))
        self.assertFalse(is_tensor([1, 2, 3]))

    def test_tensor_scatter_nd_update(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]])
        indices = paddle.to_tensor([[0, 1], [1, 0]])
        updates = paddle.to_tensor([9, 10])
        updated_tensor = tensor_scatter_nd_update(tensor, indices, updates)
        expected = np.array([[1, 9], [10, 4]])
        self.assertTrue(np.allclose(updated_tensor.numpy(), expected))

    def test_diag(self):
        tensor = paddle.to_tensor([1, 2, 3])
        diag_tensor = diag(tensor)
        expected = np.diag([1, 2, 3])
        self.assertTrue(np.allclose(diag_tensor.numpy(), expected))

    def test_mask_select(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]])
        mask = paddle.to_tensor([[True, False], [False, True]])
        selected = mask_select(tensor, mask)
        expected = np.array([1, 4])
        self.assertTrue(np.allclose(selected.numpy(), expected))

    def test_eye(self):
        tensor = eye(3)
        expected = np.eye(3)
        self.assertTrue(np.allclose(tensor.numpy(), expected))

    def test_einsum(self):
        tensor1 = paddle.to_tensor([[1, 2], [3, 4]])
        tensor2 = paddle.to_tensor([[5, 6], [7, 8]])
        result = einsum("ij,jk->ik", tensor1, tensor2)
        expected = np.einsum("ij,jk->ik", tensor1.numpy(), tensor2.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    # def test_set_device(self):
    #     set_device("cpu")
    #     self.assertEqual(get_device(), "cpu")

    # def test_to_device(self):
    #     tensor = paddle.to_tensor([1, 2, 3])
    #     tensor_on_cpu = to_device(tensor, "cpu")
    #     self.assertTrue(is_tensor(tensor_on_cpu))

    # def test_roll(self):
    #     tensor = paddle.to_tensor([1, 2, 3, 4])
    #     rolled_tensor = roll(tensor, shifts=2)
    #     expected = np.roll(tensor.numpy(), 2)
    #     self.assertTrue(np.allclose(rolled_tensor.numpy(), expected))

    def test_logsoftmax(self):
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        logsoftmax_tensor = logsoftmax(tensor)
        expected = np.log(np.exp(tensor.numpy()) / np.sum(np.exp(tensor.numpy())))
        self.assertTrue(np.allclose(logsoftmax_tensor.numpy(), expected))

    def test_topk(self):
        tensor = paddle.to_tensor([1, 3, 2])
        values, indices = topk(tensor, k=2)
        self.assertTrue(np.allclose(values.numpy(), [3, 2]))
        self.assertTrue(np.allclose(indices.numpy(), [1, 2]))

    def test_numel(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]])
        self.assertEqual(numel(tensor), 4)

    def test_histogram(self):
        tensor = paddle.to_tensor([1, 2, 1])
        hist = histogram(tensor, bins=2, min=1, max=2)
        expected = np.histogram(tensor.numpy(), bins=2, range=(1, 2))[0]
        self.assertTrue(np.allclose(hist.numpy(), expected))

    def test_flatten(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]])
        flattened = flatten(tensor)
        expected = tensor.numpy().flatten()
        self.assertTrue(np.allclose(flattened.numpy(), expected))

    def test_interpolate(self):
        tensor = paddle.to_tensor([[[[1, 2], [3, 4]]]], dtype="float32")
        interpolated = interpolate(tensor, size=[4, 4], mode="nearest")
        expected = np.array([[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]], dtype="float32")
        self.assertTrue(np.allclose(interpolated.numpy(), expected))

    def test_index_select(self):
        tensor = paddle.to_tensor([1, 2, 3, 4])
        indices = paddle.to_tensor([0, 2])
        selected = index_select(tensor, indices)
        expected = tensor.numpy()[[0, 2]]
        self.assertTrue(np.allclose(selected.numpy(), expected))

    def test_dot(self):
        tensor1 = paddle.to_tensor([1, 2, 3])
        tensor2 = paddle.to_tensor([4, 5, 6])
        result = dot(tensor1, tensor2)
        expected = np.dot(tensor1.numpy(), tensor2.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_swish(self):
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        swish = Swish()
        result = swish(tensor)
        expected = tensor.numpy() / (1 + np.exp(-tensor.numpy()))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_expand(self):
        tensor = paddle.to_tensor([1, 2, 3])
        expanded = expand(tensor, [3, 3])
        expected = np.tile(tensor.numpy(), (3, 1))
        self.assertTrue(np.allclose(expanded.numpy(), expected))

    def test_unique(self):
        tensor = paddle.to_tensor([1, 2, 2, 3])
        unique_tensor = unique(tensor)
        expected = np.unique(tensor.numpy())
        self.assertTrue(np.allclose(unique_tensor.numpy(), expected))

    def test_flip(self):
        tensor = paddle.to_tensor([1, 2, 3])
        flipped = flip(tensor, axis=0)
        expected = np.flip(tensor.numpy(), axis=0)
        self.assertTrue(np.allclose(flipped.numpy(), expected))

    def test_mv(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]])
        vec = paddle.to_tensor([1, 1])
        result = mv(tensor, vec)
        expected = np.dot(tensor.numpy(), vec.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))


if __name__ == "__main__":
    unittest.main()
