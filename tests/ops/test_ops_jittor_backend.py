import unittest

import jittor as jt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "jittor"

from tensorlayerx.backend.ops.jittor_backend import *
from tests.utils import CustomTestCase


class TestJittorBackend(CustomTestCase):

    def test_zeros(self):
        shape = (2, 3)
        tensor = zeros(shape)
        self.assertTrue(np.array_equal(tensor.numpy(), np.zeros(shape)))

    def test_ones(self):
        shape = (2, 3)
        tensor = ones(shape)
        self.assertTrue(np.array_equal(tensor.numpy(), np.ones(shape)))

    def test_constant(self):
        value = 5
        shape = (2, 3)
        tensor = constant(value, shape=shape)
        self.assertTrue(np.array_equal(tensor.numpy(), np.full(shape, value)))

    def test_random_uniform(self):
        shape = (2, 3)
        tensor = random_uniform(shape, minval=0, maxval=1)
        self.assertTrue(np.all(tensor.numpy() >= 0) and np.all(tensor.numpy() < 1))

    def test_random_normal(self):
        shape = (2, 3)
        mean = 0.0
        stddev = 1.0
        tensor = random_normal(shape, mean=mean, stddev=stddev)
        self.assertEqual(tensor.shape, shape)

    def test_truncated_normal(self):
        shape = (2, 3)
        mean = 0.0
        stddev = 1.0
        tensor = truncated_normal(shape, mean=mean, stddev=stddev)
        self.assertEqual(tensor.shape, shape)

    def test_he_normal(self):
        shape = (2, 3)
        tensor = he_normal(shape)
        self.assertEqual(tensor.shape, shape)

    def test_he_uniform(self):
        shape = (2, 3)
        tensor = he_uniform(shape)
        self.assertEqual(tensor.shape, shape)

    def test_xavier_normal(self):
        shape = (2, 3)
        tensor = xavier_normal(shape)
        self.assertEqual(tensor.shape, shape)

    def test_xavier_uniform(self):
        shape = (2, 3)
        tensor = xavier_uniform(shape)
        self.assertEqual(tensor.shape, shape)

    # def test_Variable(self):
    #     initial_value = jt.array([1, 2, 3])
    #     var = Variable(initial_value)
    #     self.assertTrue(np.array_equal(var.numpy(), initial_value.numpy()))

    def test_matmul(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[5, 6], [7, 8]])
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_add(self):
        a = jt.array([1, 2, 3])
        b = jt.array([4, 5, 6])
        result = add(a, b)
        expected = np.array([5, 7, 9])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_dtypes(self):
        self.assertEqual(dtypes("float32"), jt.float32)
        self.assertEqual(dtypes("int32"), jt.int32)

    def test_minimum(self):
        a = jt.array([1, 2, 3])
        b = jt.array([3, 2, 1])
        result = minimum(a, b)
        expected = np.array([1, 2, 1])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reshape(self):
        tensor = jt.array([[1, 2, 3], [4, 5, 6]])
        shape = (3, 2)
        result = reshape(tensor, shape)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_concat(self):
        a = jt.array([1, 2, 3])
        b = jt.array([4, 5, 6])
        result = concat([a, b], axis=0)
        expected = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_convert_to_tensor(self):
        value = [1, 2, 3]
        tensor = convert_to_tensor(value)
        self.assertTrue(np.array_equal(tensor.numpy(), np.array(value)))

    def test_sqrt(self):
        a = jt.array([1, 4, 9])
        result = sqrt(a)
        expected = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reduce_mean(self):
        a = jt.array([1, 2, 3, 4])
        result = reduce_mean(a)
        expected = np.mean([1, 2, 3, 4])
        self.assertEqual(result.numpy(), expected)

    def test_reduce_max(self):
        a = jt.array([1, 2, 3, 4])
        result = reduce_max(a)
        expected = np.max([1, 2, 3, 4])
        self.assertEqual(result.numpy(), expected)

    def test_reduce_min(self):
        a = jt.array([1, 2, 3, 4])
        result = reduce_min(a)
        expected = np.min([1, 2, 3, 4])
        self.assertEqual(result.numpy(), expected)

    def test_pad(self):
        a = jt.array([[1, 2], [3, 4]])
        paddings = ((1, 1), (2, 2))
        result = pad(a, paddings, mode="CONSTANT", constant_values=0)
        expected = np.pad(a.numpy(), paddings, mode="constant", constant_values=0)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_stack(self):
        a = jt.array([1, 2])
        b = jt.array([3, 4])
        result = stack([a, b], axis=0)
        expected = np.stack([a.numpy(), b.numpy()], axis=0)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_arange(self):
        result = arange(0, 5, 1)
        expected = np.arange(0, 5, 1)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_expand_dims(self):
        a = jt.array([1, 2, 3])
        result = expand_dims(a, axis=0)
        expected = np.expand_dims(a.numpy(), axis=0)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_tile(self):
        a = jt.array([1, 2, 3])
        multiples = [2]
        result = tile(a, multiples)
        expected = np.tile(a.numpy(), multiples)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_cast(self):
        a = jt.array([1.1, 2.2, 3.3])
        result = cast(a, dtype=jt.int32)
        expected = a.numpy().astype(np.int32)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_transpose(self):
        a = jt.array([[1, 2, 3], [4, 5, 6]])
        result = transpose(a, perm=[1, 0])
        expected = np.transpose(a.numpy(), axes=[1, 0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_clip_by_value(self):
        a = jt.array([1, 2, 3, 4, 5])
        result = clip_by_value(a, 2, 4)
        expected = np.clip(a.numpy(), 2, 4)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_split(self):
        a = jt.array([1, 2, 3, 4, 5, 6])
        result = split(a, 2, axis=0)
        expected = np.split(a.numpy(), 2, axis=0)
        for r, e in zip(result, expected):
            self.assertTrue(np.array_equal(r.numpy(), e))

    def test_floor(self):
        a = jt.array([1.1, 2.2, 3.3])
        result = floor(a)
        expected = np.floor(a.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_gather(self):
        a = jt.array([1, 2, 3, 4, 5])
        indices = jt.array([0, 2, 4])
        result = gather(a, indices)
        expected = a.numpy()[indices.numpy()]
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_linspace(self):
        result = linspace(0, 10, 5)
        expected = np.linspace(0, 10, 5)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_slice(self):
        a = jt.array([1, 2, 3, 4, 5])
        result = slice(a, [1], [3])
        expected = a.numpy()[1:4]
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_add_n(self):
        a = jt.array([1, 2, 3])
        b = jt.array([4, 5, 6])
        result = add_n([a, b])
        expected = a.numpy() + b.numpy()
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_one_hot(self):
        a = jt.array([0, 1, 2])
        depth = 3
        result = OneHot(depth)(a)
        expected = np.eye(depth)[a.numpy()]
        self.assertTrue(np.array_equal(result.numpy(), expected))


class TestJittorBackend(unittest.TestCase):

    def test_l2_normalize(self):
        x = jt.array([[1.0, 2.0], [3.0, 4.0]])
        l2_norm = L2Normalize(axis=1)
        result = l2_norm(x)
        expected = np.array([[0.4472136, 0.8944272], [0.6, 0.8]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_embedding_lookup(self):
        params = jt.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ids = jt.array([0, 2])
        embedding_lookup = EmbeddingLookup()
        result = embedding_lookup(params, ids)
        expected = np.array([[0.1, 0.2], [0.5, 0.6]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_not_equal(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        not_equal_op = NotEqual()
        result = not_equal_op(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_count_nonzero(self):
        x = jt.array([[0, 1, 0], [2, 3, 0]])
        count_nonzero_op = CountNonzero()
        result = count_nonzero_op(x)
        expected = np.array(3)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_resize(self):
        x = jt.array(np.random.rand(1, 3, 4, 4).astype(np.float32))
        resize_op = Resize(scale=(2, 2), method="nearest", data_format="channels_first")
        result = resize_op(x)
        expected = np.repeat(np.repeat(x.numpy(), 2, axis=2), 2, axis=3)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    # def test_zero_padding_1d(self):
    #     x = jt.array(np.random.rand(1, 3, 4).astype(np.float32))
    #     zero_padding_1d = ZeroPadding1D(padding=(1, 1), data_format="channels_last")
    #     result = zero_padding_1d(x)
    #     expected = np.pad(x.numpy(), ((0, 0), (1, 1), (0, 0)), mode="constant")
    #     np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    # def test_zero_padding_2d(self):
    #     x = jt.array(np.random.rand(1, 3, 4, 4).astype(np.float32))
    #     zero_padding_2d = ZeroPadding2D(padding=(1, 1), data_format="channels_last")
    #     result = zero_padding_2d(x)
    #     expected = np.pad(x.numpy(), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    #     np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    # def test_zero_padding_3d(self):
    #     x = jt.array(np.random.rand(1, 3, 4, 4, 4).astype(np.float32))
    #     zero_padding_3d = ZeroPadding3D(padding=(1, 1, 1), data_format="channels_last")
    #     result = zero_padding_3d(x)
    #     expected = np.pad(x.numpy(), ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode="constant")
    #     np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_sign(self):
        x = jt.array([-1.0, 0.0, 1.0])
        sign_op = Sign()
        result = sign_op(x)
        expected = np.sign(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ceil(self):
        x = jt.array([1.1, 2.5, 3.7])
        ceil_op = Ceil()
        result = ceil_op(x)
        expected = np.ceil(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply(self):
        x = jt.array([1, 2, 3])
        y = jt.array([4, 5, 6])
        result = multiply(x, y)
        expected = np.multiply(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_divide(self):
        x = jt.array([4, 5, 6])
        y = jt.array([2, 2, 2])
        result = divide(x, y)
        expected = np.divide(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_triu(self):
        x = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = triu(x)
        expected = np.triu(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_tril(self):
        x = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = tril(x)
        expected = np.tril(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_abs(self):
        x = jt.array([-1, -2, 3])
        result = abs(x)
        expected = np.abs(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_acos(self):
        x = jt.array([1, 0, -1])
        result = acos(x)
        expected = np.arccos(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_acosh(self):
        x = jt.array([1, 2, 3])
        result = acosh(x)
        expected = np.arccosh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_argmax(self):
        x = jt.array([[1, 2, 3], [4, 5, 6]])
        result = argmax(x, axis=1)
        expected = np.argmax(x.numpy(), axis=1)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_argmin(self):
        x = jt.array([[1, 2, 3], [4, 5, 6]])
        result = argmin(x, axis=1)
        expected = np.argmin(x.numpy(), axis=1)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_asin(self):
        x = jt.array([1, 0, -1])
        result = asin(x)
        expected = np.arcsin(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_asinh(self):
        x = jt.array([1, 2, 3])
        result = asinh(x)
        expected = np.arcsinh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_atan(self):
        x = jt.array([1, 0, -1])
        result = atan(x)
        expected = np.arctan(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_atanh(self):
        x = jt.array([0.5, 0, -0.5])
        result = atanh(x)
        expected = np.arctanh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_cos(self):
        x = jt.array([0, np.pi / 2, np.pi])
        result = cos(x)
        expected = np.cos(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_cosh(self):
        x = jt.array([0, 1, -1])
        result = cosh(x)
        expected = np.cosh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_cumprod(self):
        x = jt.array([1, 2, 3, 4])
        result = cumprod(x)
        expected = np.cumprod(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cumsum(self):
        x = jt.array([1, 2, 3, 4])
        result = cumsum(x)
        expected = np.cumsum(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = equal(x, y)
        expected = np.equal(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_exp(self):
        x = jt.array([1, 2, 3])
        result = exp(x)
        expected = np.exp(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_floordiv(self):
        x = jt.array([4, 5, 6])
        y = jt.array([2, 2, 2])
        result = floordiv(x, y)
        expected = np.floor_divide(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = greater(x, y)
        expected = np.greater(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = greater_equal(x, y)
        expected = np.greater_equal(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_is_inf(self):
        x = jt.array([1, np.inf, -np.inf])
        result = is_inf(x)
        expected = np.isinf(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_is_nan(self):
        x = jt.array([1, np.nan, 3])
        result = is_nan(x)
        expected = np.isnan(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_l2_normalize_func(self):
        x = jt.array([[1.0, 2.0], [3.0, 4.0]])
        result = l2_normalize(x, axis=1)
        expected = np.array([[0.4472136, 0.8944272], [0.6, 0.8]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_less(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = less(x, y)
        expected = np.less(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = less_equal(x, y)
        expected = np.less_equal(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_log(self):
        x = jt.array([1, 2, 3])
        result = log(x)
        expected = np.log(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_log_sigmoid(self):
        x = jt.array([1, 2, 3])
        result = log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-x.numpy())))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_maximum(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = maximum(x, y)
        expected = np.maximum(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_negative(self):
        x = jt.array([1, 2, 3])
        result = negative(x)
        expected = np.negative(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_func(self):
        x = jt.array([1, 2, 3])
        y = jt.array([1, 0, 3])
        result = not_equal(x, y)
        expected = np.not_equal(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_pow(self):
        x = jt.array([1, 2, 3])
        y = jt.array([2, 2, 2])
        result = pow(x, y)
        expected = np.power(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_real(self):
        x = jt.array([1, 2, 3])
        result = real(x)
        expected = x.numpy()
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reciprocal(self):
        x = jt.array([1, 2, 3])
        result = reciprocal(x)
        expected = np.reciprocal(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_prod(self):
        x = jt.array([1, 2, 3])
        result = reduce_prod(x)
        expected = np.prod(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_std(self):
        x = jt.array([1, 2, 3])
        result = reduce_std(x)
        expected = np.std(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_reduce_sum(self):
        x = jt.array([1, 2, 3])
        result = reduce_sum(x)
        expected = np.sum(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reduce_variance(self):
        x = jt.array([1, 2, 3])
        result = reduce_variance(x)
        expected = np.var(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_round(self):
        x = jt.array([1.1, 2.5, 3.7])
        result = round(x)
        expected = np.round(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_rsqrt(self):
        x = jt.array([1, 4, 9])
        result = rsqrt(x)
        expected = 1 / np.sqrt(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_max(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        result = segment_max(x, segment_ids)
        expected = np.array([2, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_segment_mean(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        result = segment_mean(x, segment_ids)
        expected = np.array([1.5, 3.5])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_min(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        result = segment_min(x, segment_ids)
        expected = np.array([1, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_segment_sum(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        result = segment_sum(x, segment_ids)
        expected = np.array([3, 7])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_sigmoid(self):
        x = jt.array([0.0, 1.0, -1.0])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-x.numpy()))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_sign_func(self):
        x = jt.array([-1.0, 0.0, 1.0])
        result = sign(x)
        expected = np.sign(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_sin(self):
        x = jt.array([0.0, np.pi / 2, np.pi])
        result = sin(x)
        expected = np.sin(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_sinh(self):
        x = jt.array([0.0, 1.0, -1.0])
        result = sinh(x)
        expected = np.sinh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_softplus(self):
        x = jt.array([0.0, 1.0, -1.0])
        result = softplus(x)
        expected = np.log1p(np.exp(x.numpy()))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_square(self):
        x = jt.array([1.0, 2.0, -3.0])
        result = square(x)
        expected = np.square(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_squared_difference(self):
        x = jt.array([1.0, 2.0, 3.0])
        y = jt.array([1.0, 0.0, 3.0])
        result = squared_difference(x, y)
        expected = np.square(x.numpy() - y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_subtract(self):
        x = jt.array([1.0, 2.0, 3.0])
        y = jt.array([1.0, 0.0, 3.0])
        result = subtract(x, y)
        expected = np.subtract(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_tan(self):
        x = jt.array([0.0, np.pi / 4, np.pi / 2])
        result = tan(x)
        expected = np.tan(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_tanh(self):
        x = jt.array([0.0, 1.0, -1.0])
        result = tanh(x)
        expected = np.tanh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_any(self):
        x = jt.array([0, 1, 0])
        result = any(x)
        expected = np.any(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_all(self):
        x = jt.array([1, 1, 1])
        result = all(x)
        expected = np.all(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_and(self):
        x = jt.array([True, False, True])
        y = jt.array([True, True, False])
        result = logical_and(x, y)
        expected = np.logical_and(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_or(self):
        x = jt.array([True, False, True])
        y = jt.array([True, True, False])
        result = logical_or(x, y)
        expected = np.logical_or(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_not(self):
        x = jt.array([True, False, True])
        result = logical_not(x)
        expected = np.logical_not(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_xor(self):
        x = jt.array([True, False, True])
        y = jt.array([True, True, False])
        result = logical_xor(x, y)
        expected = np.logical_xor(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_argsort(self):
        x = jt.array([3, 1, 2])
        result = argsort(x)
        expected = np.argsort(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_bmm(self):
        x = jt.array([[[1, 2], [3, 4]]])
        y = jt.array([[[5, 6], [7, 8]]])
        result = bmm(x, y)
        expected = np.matmul(x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_where(self):
        condition = jt.array([True, False, True])
        x = jt.array([1, 2, 3])
        y = jt.array([4, 5, 6])
        result = where(condition, x, y)
        expected = np.where(condition.numpy(), x.numpy(), y.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones_like(self):
        x = jt.array([1, 2, 3])
        result = ones_like(x)
        expected = np.ones_like(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_zeros_like(self):
        x = jt.array([1, 2, 3])
        result = zeros_like(x)
        expected = np.zeros_like(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_squeeze(self):
        x = jt.array([[[1, 2, 3]]])
        result = squeeze(x)
        expected = np.squeeze(x.numpy())
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unsorted_segment_sum(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        num_segments = 2
        result = unsorted_segment_sum(x, segment_ids, num_segments)
        expected = np.array([3, 7])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unsorted_segment_mean(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        num_segments = 2
        result = unsorted_segment_mean(x, segment_ids, num_segments)
        expected = np.array([1.5, 3.5])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_unsorted_segment_min(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        num_segments = 2
        result = unsorted_segment_min(x, segment_ids, num_segments)
        expected = np.array([1, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unsorted_segment_max(self):
        x = jt.array([1, 2, 3, 4])
        segment_ids = jt.array([0, 0, 1, 1])
        num_segments = 2
        result = unsorted_segment_max(x, segment_ids, num_segments)
        expected = np.array([2, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_set_seed(self):
        set_seed(42)
        a = random_normal((2, 3))
        set_seed(42)
        b = random_normal((2, 3))
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_is_tensor(self):
        a = jt.array([1, 2, 3])
        self.assertTrue(is_tensor(a))
        self.assertFalse(is_tensor([1, 2, 3]))

    def test_diag(self):
        a = jt.array([1, 2, 3])
        result = diag(a)
        expected = np.diag(a.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_eye(self):
        result = eye(3)
        expected = np.eye(3)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_einsum(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[5, 6], [7, 8]])
        result = einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", a.numpy(), b.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))

    # def test_get_device(self):
    #     device = get_device()
    #     self.assertIn(device, ["CPU", "GPU:0"])

    # def test_roll(self):
    #     a = jt.array([1, 2, 3, 4, 5])
    #     result = roll(a, 2)
    #     expected = np.roll(a.numpy(), 2)
    #     self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logsoftmax(self):
        a = jt.array([1, 2, 3])
        result = logsoftmax(a)
        expected = np.log(np.exp(a.numpy()) / np.sum(np.exp(a.numpy())))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_topk(self):
        a = jt.array([1, 3, 2, 4, 5])
        result, indices = topk(a, 3)
        expected_values = np.array([5, 4, 3])
        expected_indices = np.array([4, 3, 1])
        self.assertTrue(np.array_equal(result.numpy(), expected_values))
        self.assertTrue(np.array_equal(indices.numpy(), expected_indices))

    def test_numel(self):
        a = jt.array([1, 2, 3, 4, 5])
        result = numel(a)
        expected = np.size(a.numpy())
        self.assertEqual(result, expected)

    def test_flip(self):
        a = jt.array([1, 2, 3])
        result = flip(a, axis=0)
        expected = np.flip(a.numpy(), axis=0)
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_mv(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([5, 6])
        result = mv(a, b)
        expected = np.matmul(a.numpy(), b.numpy())
        self.assertTrue(np.array_equal(result.numpy(), expected))


if __name__ == "__main__":
    unittest.main()
