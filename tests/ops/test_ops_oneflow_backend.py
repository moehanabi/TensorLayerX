import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "oneflow"
import numpy as np
import oneflow as flow

from tensorlayerx.backend.ops.oneflow_backend import *
from tests.utils import CustomTestCase


class TestOneFlowBackend(CustomTestCase):

    def test_get_tensor_shape(self):
        x = flow.ones((2, 3, 4))
        self.assertEqual(get_tensor_shape(x), [2, 3, 4])

    def test_zeros(self):
        x = zeros((2, 3), dtype="float32")
        self.assertTrue(np.array_equal(x.numpy(), np.zeros((2, 3))))

    def test_ones(self):
        x = ones((2, 3), dtype="float32")
        self.assertTrue(np.array_equal(x.numpy(), np.ones((2, 3))))

    def test_constant(self):
        x = constant(5, (2, 3), dtype="int32")
        self.assertTrue(np.array_equal(x.numpy(), np.full((2, 3), 5)))

    def test_random_uniform(self):
        x = random_uniform((2, 3), minval=0, maxval=1, dtype="float32")
        self.assertTrue(np.all(x.numpy() >= 0) and np.all(x.numpy() < 1))

    def test_random_normal(self):
        x = random_normal((2, 3), mean=0.0, stddev=1.0, dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_truncated_normal(self):
        x = truncated_normal((2, 3), mean=0.0, stddev=1.0, dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_he_normal(self):
        x = he_normal((2, 3), dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_he_uniform(self):
        x = he_uniform((2, 3), dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_xavier_normal(self):
        x = xavier_normal((2, 3), dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_xavier_uniform(self):
        x = xavier_uniform((2, 3), dtype="float32")
        self.assertEqual(x.shape, (2, 3))

    def test_Variable(self):
        x = Variable(flow.ones((2, 3)), name="var", trainable=True)
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue(x.requires_grad)

    def test_matmul(self):
        a = flow.ones((2, 3))
        b = flow.ones((3, 2))
        c = matmul(a, b)
        self.assertTrue(np.array_equal(c.numpy(), np.full((2, 2), 3)))

    def test_add(self):
        a = flow.ones((2, 3))
        b = flow.ones((2, 3))
        c = add(a, b)
        self.assertTrue(np.array_equal(c.numpy(), np.full((2, 3), 2)))

    def test_dtypes(self):
        self.assertEqual(dtypes("float32"), flow.float32)

    def test_maximum(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([3, 2, 1])
        c = maximum(a, b)
        self.assertTrue(np.array_equal(c.numpy(), [3, 2, 3]))

    def test_minimum(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([3, 2, 1])
        c = minimum(a, b)
        self.assertTrue(np.array_equal(c.numpy(), [1, 2, 1]))

    def test_reshape(self):
        a = flow.ones((2, 3))
        b = reshape(a, (3, 2))
        self.assertEqual(b.shape, (3, 2))

    def test_concat(self):
        a = flow.ones((2, 3))
        b = flow.ones((2, 3))
        c = concat([a, b], axis=0)
        self.assertEqual(c.shape, (4, 3))

    def test_convert_to_tensor(self):
        a = convert_to_tensor([1, 2, 3], dtype="float32")
        self.assertTrue(np.array_equal(a.numpy(), [1, 2, 3]))

    def test_convert_to_numpy(self):
        a = flow.ones((2, 3))
        b = convert_to_numpy(a)
        self.assertTrue(np.array_equal(b, np.ones((2, 3))))

    def test_sqrt(self):
        a = flow.tensor([4.0, 9.0, 16.0])
        b = sqrt(a)
        self.assertTrue(np.array_equal(b.numpy(), [2.0, 3.0, 4.0]))

    def test_reduce_mean(self):
        a = flow.ones((2, 3))
        b = reduce_mean(a)
        self.assertEqual(b.numpy(), 1.0)

    def test_reduce_max(self):
        a = flow.tensor([1, 2, 3])
        b = reduce_max(a)
        self.assertEqual(b.numpy(), 3)

    def test_reduce_min(self):
        a = flow.tensor([1, 2, 3])
        b = reduce_min(a)
        self.assertEqual(b.numpy(), 1)

    def test_pad(self):
        a = flow.ones((2, 2))
        b = pad(a, paddings=[[1, 1], [2, 2]], mode="CONSTANT", constant_values=0)
        self.assertTrue(np.array_equal(b.numpy(), np.pad(np.ones((2, 2)), ((1, 1), (2, 2)), "constant")))

    def test_stack(self):
        a = flow.ones((2, 2))
        b = flow.ones((2, 2))
        c = stack([a, b], axis=0)
        self.assertEqual(c.shape, (2, 2, 2))

    def test_meshgrid(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([4, 5, 6])
        c, d = meshgrid(a, b)
        self.assertTrue(np.array_equal(c.numpy(), np.meshgrid([1, 2, 3], [4, 5, 6])[0]))
        self.assertTrue(np.array_equal(d.numpy(), np.meshgrid([1, 2, 3], [4, 5, 6])[1]))

    def test_arange(self):
        a = arange(0, 5, 1)
        self.assertTrue(np.array_equal(a.numpy(), np.arange(0, 5, 1)))

    def test_expand_dims(self):
        a = flow.ones((2, 2))
        b = expand_dims(a, axis=0)
        self.assertEqual(b.shape, (1, 2, 2))

    def test_tile(self):
        a = flow.ones((2, 2))
        b = tile(a, multiples=(2, 2))
        self.assertEqual(b.shape, (4, 4))

    def test_cast(self):
        a = flow.ones((2, 2), dtype=flow.float32)
        b = cast(a, dtype=flow.int32)
        self.assertEqual(b.dtype, flow.int32)

    def test_transpose(self):
        a = flow.ones((2, 3, 4))
        b = transpose(a, perm=[1, 0, 2])
        self.assertEqual(b.shape, (3, 2, 4))

    def test_gather_nd(self):
        params = flow.tensor([[1, 2], [3, 4]])
        indices = flow.tensor([[0, 0], [1, 1]])
        b = gather_nd(params, indices)
        self.assertTrue(np.array_equal(b.numpy(), [1, 4]))

    def test_scatter_nd(self):
        indices = flow.tensor([[0], [2], [4]])
        updates = flow.tensor([1, 2, 3])
        shape = flow.Size([6])
        b = scatter_nd(indices, updates, shape)
        self.assertTrue(np.array_equal(b.numpy(), [1, 0, 2, 0, 3, 0]))

    def test_clip_by_value(self):
        a = flow.tensor([1, 2, 3, 4, 5])
        b = clip_by_value(a, clip_value_min=2, clip_value_max=4)
        self.assertTrue(np.array_equal(b.numpy(), [2, 2, 3, 4, 4]))

    def test_split(self):
        a = flow.ones((6,))
        b = split(a, num_or_size_splits=3, axis=0)
        self.assertEqual(len(b), 3)
        self.assertTrue(np.array_equal(b[0].numpy(), np.ones((2,))))

    def test_floor(self):
        a = flow.tensor([1.2, 2.5, 3.7])
        b = floor(a)
        self.assertTrue(np.array_equal(b.numpy(), [1.0, 2.0, 3.0]))

    def test_gather(self):
        params = flow.tensor([1, 2, 3, 4, 5])
        indices = flow.tensor([0, 2, 4])
        b = gather(params, indices)
        self.assertTrue(np.array_equal(b.numpy(), [1, 3, 5]))

    def test_linspace(self):
        a = linspace(0, 10, 5)
        self.assertTrue(np.array_equal(a.numpy(), np.linspace(0, 10, 5)))

    def test_slice(self):
        a = flow.tensor([1, 2, 3, 4, 5])
        b = slice(a, starts=[1], sizes=[3])
        self.assertTrue(np.array_equal(b.numpy(), [2, 3, 4]))

    def test_add_n(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([4, 5, 6])
        c = add_n([a, b])
        self.assertTrue(np.array_equal(c.numpy(), [5, 7, 9]))

    def test_one_hot(self):
        a = flow.tensor([0, 1, 2])
        b = OneHot(depth=3)(a)
        self.assertTrue(np.array_equal(b.numpy(), np.eye(3)[[0, 1, 2]]))

    def test_l2_normalize(self):
        a = flow.tensor([1.0, 2.0, 3.0])
        b = L2Normalize(axis=0)(a)
        self.assertTrue(np.allclose(b.numpy(), a.numpy() / np.linalg.norm(a.numpy())))

    def test_embedding_lookup(self):
        params = flow.tensor([[1, 2], [3, 4], [5, 6]])
        ids = flow.tensor([0, 2])
        b = EmbeddingLookup()(params, ids)
        self.assertTrue(np.array_equal(b.numpy(), [[1, 2], [5, 6]]))

    def test_not_equal(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([3, 2, 1])
        c = NotEqual()(a, b)
        self.assertTrue(np.array_equal(c.numpy(), [True, False, True]))

    def test_count_nonzero(self):
        a = flow.tensor([0, 1, 2, 0, 3])
        b = CountNonzero()(a)
        self.assertEqual(b.numpy(), 3)

    def test_resize(self):
        a = flow.ones((1, 1, 2, 2))
        b = Resize(scale=2, method="nearest")(a)
        self.assertEqual(b.shape, (1, 1, 4, 4))

    def test_zero_padding_1d(self):
        a = flow.ones((1, 2, 3))
        b = ZeroPadding1D(padding=(1, 1), data_format="channels_last")(a)
        self.assertEqual(b.shape, (1, 4, 3))

    def test_zero_padding_2d(self):
        a = flow.ones((1, 2, 2, 3))
        b = ZeroPadding2D(padding=(1, 1), data_format="channels_last")(a)
        self.assertEqual(b.shape, (1, 4, 4, 3))

    def test_zero_padding_3d(self):
        a = flow.ones((1, 2, 2, 2, 3))
        b = ZeroPadding3D(padding=(1, 1, 1), data_format="channels_last")(a)
        self.assertEqual(b.shape, (1, 4, 4, 4, 3))

    def test_ceil(self):
        a = flow.tensor([1.2, 2.5, 3.7])
        b = ceil(a)
        self.assertTrue(np.array_equal(b.numpy(), [2.0, 3.0, 4.0]))

    def test_multiply(self):
        a = flow.tensor([1, 2, 3])
        b = flow.tensor([4, 5, 6])
        c = multiply(a, b)
        self.assertTrue(np.array_equal(c.numpy(), [4, 10, 18]))

    def test_divide(self):
        a = flow.tensor([4, 5, 6])
        b = flow.tensor([2, 2, 2])
        c = divide(a, b)
        self.assertTrue(np.array_equal(c.numpy(), [2, 2.5, 3]))

    def test_triu(self):
        a = flow.ones((3, 3))
        b = triu(a)
        self.assertTrue(np.array_equal(b.numpy(), np.triu(np.ones((3, 3)))))

    def test_tril(self):
        a = flow.ones((3, 3))
        b = tril(a)
        self.assertTrue(np.array_equal(b.numpy(), np.tril(np.ones((3, 3)))))

    def test_abs(self):
        a = flow.tensor([-1, -2, -3])
        b = abs(a)
        self.assertTrue(np.array_equal(b.numpy(), [1, 2, 3]))

    def test_acos(self):
        a = flow.tensor([1.0, 0.0, -1.0])
        b = acos(a)
        self.assertTrue(np.allclose(b.numpy(), np.arccos([1.0, 0.0, -1.0])))

    def test_acosh(self):
        a = flow.tensor([1.0, 2.0, 3.0])
        b = acosh(a)
        self.assertTrue(np.allclose(b.numpy(), np.arccosh([1.0, 2.0, 3.0])))

    def test_argmax(self):
        a = flow.tensor([1, 3, 2])
        b = argmax(a)
        self.assertEqual(b.numpy(), 1)

    def test_argmin(self):
        a = flow.tensor([1, 3, 2])
        b = argmin(a)
        self.assertEqual(b.numpy(), 0)

    def test_asin(self):
        a = flow.tensor([0.0, 0.5, 1.0])
        b = asin(a)
        self.assertTrue(np.allclose(b.numpy(), np.arcsin([0.0, 0.5, 1.0])))

    def test_asinh(self):
        a = flow.tensor([0.0, 0.5, 1.0])
        b = asinh(a)
        self.assertTrue(np.allclose(b.numpy(), np.arcsinh([0.0, 0.5, 1.0])))

    def test_atan(self):
        a = flow.tensor([0.0, 0.5, 1.0])
        b = atan(a)
        self.assertTrue(np.allclose(b.numpy(), np.arctan([0.0, 0.5, 1.0])))

    def test_atanh(self):
        a = flow.tensor([0.0, 0.5, 1.0])
        b = atanh(a)
        self.assertTrue(np.allclose(b.numpy(), np.arctanh([0.0, 0.5, 1.0])))

    def test_cos(self):
        a = flow.tensor([0.0, np.pi / 2, np.pi])
        b = cos(a)
        self.assertTrue(np.allclose(b.numpy(), np.cos([0.0, np.pi / 2, np.pi])))

    def test_cosh(self):
        x = flow.tensor([0.0, 1.0, 2.0])
        result = cosh(x)
        expected = np.cosh([0.0, 1.0, 2.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_count_nonzero(self):
        x = flow.tensor([[0, 1, 2], [3, 4, 0]])
        result = count_nonzero(x)
        expected = 4
        self.assertEqual(result.numpy(), expected)

    def test_cumprod(self):
        x = flow.tensor([1, 2, 3, 4])
        result = cumprod(x)
        expected = np.cumprod([1, 2, 3, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_cumsum(self):
        x = flow.tensor([1, 2, 3, 4])
        result = cumsum(x)
        expected = np.cumsum([1, 2, 3, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_equal(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 2, 4])
        result = equal(x, y)
        expected = np.equal([1, 2, 3], [1, 2, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_exp(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = exp(x)
        expected = np.exp([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_floordiv(self):
        x = flow.tensor([4, 7, 9])
        y = flow.tensor([2, 3, 4])
        result = floordiv(x, y)
        expected = np.floor_divide([4, 7, 9], [2, 3, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_greater(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 1, 4])
        result = greater(x, y)
        expected = np.greater([1, 2, 3], [1, 1, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_greater_equal(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 2, 4])
        result = greater_equal(x, y)
        expected = np.greater_equal([1, 2, 3], [1, 2, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_is_inf(self):
        x = flow.tensor([1.0, float("inf"), 2.0])
        result = is_inf(x)
        expected = np.isinf([1.0, float("inf"), 2.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_is_nan(self):
        x = flow.tensor([1.0, float("nan"), 2.0])
        result = is_nan(x)
        expected = np.isnan([1.0, float("nan"), 2.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_l2_normalize(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = l2_normalize(x)
        expected = x.numpy() / np.linalg.norm(x.numpy(), ord=2)
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_less(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 3, 2])
        result = less(x, y)
        expected = np.less([1, 2, 3], [1, 3, 2])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_less_equal(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 2, 4])
        result = less_equal(x, y)
        expected = np.less_equal([1, 2, 3], [1, 2, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_log(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = log(x)
        expected = np.log([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_log_sigmoid(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-np.array([1.0, 2.0, 3.0]))))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_negative(self):
        x = flow.tensor([1.0, -2.0, 3.0])
        result = negative(x)
        expected = np.negative([1.0, -2.0, 3.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_not_equal(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([1, 2, 4])
        result = not_equal(x, y)
        expected = np.not_equal([1, 2, 3], [1, 2, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_pow(self):
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([2, 2, 2])
        result = pow(x, y)
        expected = np.power([1, 2, 3], [2, 2, 2])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reciprocal(self):
        x = flow.tensor([1.0, 2.0, 4.0])
        result = reciprocal(x)
        expected = np.reciprocal([1.0, 2.0, 4.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_prod(self):
        x = flow.tensor([1, 2, 3, 4])
        result = reduce_prod(x)
        expected = np.prod([1, 2, 3, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reduce_std(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduce_std(x)
        expected = np.std([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_reduce_sum(self):
        x = flow.tensor([1, 2, 3, 4])
        result = reduce_sum(x)
        expected = np.sum([1, 2, 3, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_reduce_variance(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduce_variance(x)
        expected = np.var([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_round(self):
        x = flow.tensor([1.2, 2.5, 3.7])
        result = round(x)
        expected = np.round([1.2, 2.5, 3.7])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_rsqrt(self):
        x = flow.tensor([1.0, 4.0, 9.0])
        result = rsqrt(x)
        expected = 1 / np.sqrt([1.0, 4.0, 9.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_sigmoid(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_sign(self):
        x = flow.tensor([1.0, -2.0, 0.0])
        result = sign(x)
        expected = np.sign([1.0, -2.0, 0.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_sin(self):
        x = flow.tensor([0.0, np.pi / 2, np.pi])
        result = sin(x)
        expected = np.sin([0.0, np.pi / 2, np.pi])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_sinh(self):
        x = flow.tensor([0.0, 1.0, 2.0])
        result = sinh(x)
        expected = np.sinh([0.0, 1.0, 2.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_softplus(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = softplus(x)
        expected = np.log(1 + np.exp([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_square(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        result = square(x)
        expected = np.square([1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_squared_difference(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        y = flow.tensor([1.0, 1.0, 1.0])
        result = squared_difference(x, y)
        expected = np.square(np.array([1.0, 2.0, 3.0]) - np.array([1.0, 1.0, 1.0]))
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_subtract(self):
        x = flow.tensor([1.0, 2.0, 3.0])
        y = flow.tensor([1.0, 1.0, 1.0])
        result = subtract(x, y)
        expected = np.subtract([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_tan(self):
        x = flow.tensor([0.0, np.pi / 4, np.pi / 2])
        result = tan(x)
        expected = np.tan([0.0, np.pi / 4, np.pi / 2])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_tanh(self):
        x = flow.tensor([0.0, 1.0, 2.0])
        result = tanh(x)
        expected = np.tanh([0.0, 1.0, 2.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_any(self):
        x = flow.tensor([0, 1, 2, 3])
        result = any(x)
        expected = np.any([0, 1, 2, 3])
        self.assertEqual(result.numpy(), expected)

    def test_all(self):
        x = flow.tensor([1, 1, 1, 1])
        result = all(x)
        expected = np.all([1, 1, 1, 1])
        self.assertEqual(result.numpy(), expected)

    def test_logical_and(self):
        x = flow.tensor([True, False, True])
        y = flow.tensor([True, True, False])
        result = logical_and(x, y)
        expected = np.logical_and([True, False, True], [True, True, False])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_or(self):
        x = flow.tensor([True, False, True])
        y = flow.tensor([True, True, False])
        result = logical_or(x, y)
        expected = np.logical_or([True, False, True], [True, True, False])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_not(self):
        x = flow.tensor([True, False, True])
        result = logical_not(x)
        expected = np.logical_not([True, False, True])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_logical_xor(self):
        x = flow.tensor([True, False, True])
        y = flow.tensor([True, True, False])
        result = logical_xor(x, y)
        expected = np.logical_xor([True, False, True], [True, True, False])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_argsort(self):
        x = flow.tensor([3, 1, 2])
        result = argsort(x)
        expected = np.argsort([3, 1, 2])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_bmm(self):
        x = flow.randn(10, 3, 4)
        y = flow.randn(10, 4, 5)
        result = bmm(x, y)
        expected = np.matmul(x.numpy(), y.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_where(self):
        condition = flow.tensor([True, False, True])
        x = flow.tensor([1, 2, 3])
        y = flow.tensor([4, 5, 6])
        result = where(condition, x, y)
        expected = np.where([True, False, True], [1, 2, 3], [4, 5, 6])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_ones_like(self):
        x = flow.tensor([1, 2, 3])
        result = ones_like(x)
        expected = np.ones_like([1, 2, 3])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_zeros_like(self):
        x = flow.tensor([1, 2, 3])
        result = zeros_like(x)
        expected = np.zeros_like([1, 2, 3])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_squeeze(self):
        x = flow.tensor([[[1], [2], [3]]])
        result = squeeze(x)
        expected = np.squeeze([[[1], [2], [3]]])
        self.assertTrue(np.array_equal(result.numpy(), expected))

    def test_unsorted_segment_mean(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        segment_ids = flow.tensor([0, 0, 1, 1])
        result = unsorted_segment_mean(x, segment_ids, 2)
        expected = np.array([1.5, 3.5])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_unsorted_segment_sum(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        segment_ids = flow.tensor([0, 0, 1, 1])
        result = unsorted_segment_sum(x, segment_ids, 2)
        expected = np.array([3.0, 7.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_unsorted_segment_max(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        segment_ids = flow.tensor([0, 0, 1, 1])
        result = unsorted_segment_max(x, segment_ids, 2)
        expected = np.array([2.0, 4.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_unsorted_segment_min(self):
        x = flow.tensor([1.0, 2.0, 3.0, 4.0])
        segment_ids = flow.tensor([0, 0, 1, 1])
        result = unsorted_segment_min(x, segment_ids, 2)
        expected = np.array([1.0, 3.0])
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_set_seed(self):
        set_seed(42)
        self.assertEqual(np.random.randint(0, 100), 51)
        self.assertEqual(random.randint(0, 100), 81)

    def test_is_tensor(self):
        tensor = flow.Tensor([1, 2, 3])
        self.assertTrue(is_tensor(tensor))
        self.assertFalse(is_tensor([1, 2, 3]))

    def test_tensor_scatter_nd_update(self):
        tensor = flow.Tensor([1, 2, 3, 4])
        indices = [[1], [3]]
        updates = [9, 10]
        updated_tensor = tensor_scatter_nd_update(tensor, indices, updates)
        self.assertTrue(np.array_equal(updated_tensor.numpy(), [1, 9, 3, 10]))

    def test_diag(self):
        input = flow.Tensor([1, 2, 3])
        result = diag(input)
        expected = flow.Tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_mask_select(self):
        x = flow.Tensor([[1, 2], [3, 4]])
        mask = flow.Tensor([[True, False], [False, True]])
        result = mask_select(x, mask)
        expected = flow.Tensor([1, 4])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_eye(self):
        result = eye(3)
        expected = flow.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_einsum(self):
        x = flow.Tensor([[1, 2], [3, 4]])
        y = flow.Tensor([[2, 0], [1, 3]])
        result = einsum("ij,jk->ik", x, y)
        expected = flow.Tensor([[4, 6], [10, 12]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_set_device(self):
        set_device("GPU", 0)
        self.assertEqual(get_device(), "GPU:0")

    def test_get_device(self):
        device = get_device()
        self.assertIn(device, ["CPU", "GPU:0"])

    def test_to_device(self):
        tensor = flow.Tensor([1, 2, 3])
        result = to_device(tensor, "GPU", 0)
        self.assertEqual(result.device, flow.device("cuda:0"))

    def test_roll(self):
        input = flow.Tensor([1, 2, 3, 4])
        result = roll(input, shifts=2)
        expected = flow.Tensor([3, 4, 1, 2])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_logsoftmax(self):
        input = flow.Tensor([1.0, 2.0, 3.0])
        result = logsoftmax(input, dim=0)
        expected = flow.Tensor([-2.407606, -1.407606, -0.407606])
        self.assertTrue(np.allclose(result.numpy(), expected.numpy(), atol=1e-6))

    def test_topk(self):
        input = flow.Tensor([1, 3, 2, 4])
        values, indices = topk(input, k=2)
        self.assertTrue(np.array_equal(values.numpy(), [4, 3]))
        self.assertTrue(np.array_equal(indices.numpy(), [3, 1]))

    def test_numel(self):
        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        result = numel(input)
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
