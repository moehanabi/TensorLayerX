import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "mindspore"
import numpy as np
from mindspore import Tensor

from tensorlayerx.backend.ops.mindspore_backend import *
from tests.utils import CustomTestCase

set_device(device="CPU", id=0)


class TestMindsporeBackend(CustomTestCase):

    def test_zeros(self):
        shape = (2, 3)
        result = zeros(shape)
        expected = np.zeros(shape)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_ones(self):
        shape = (2, 3)
        result = ones(shape)
        expected = np.ones(shape)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_constant(self):
        value = 5
        shape = (2, 3)
        result = constant(value, shape=shape)
        expected = np.full(shape, value)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_random_uniform(self):
        shape = (2, 3)
        result = random_uniform(shape, minval=0, maxval=1)
        self.assertEqual(result.shape, shape)

    def test_random_normal(self):
        shape = (2, 3)
        result = random_normal(shape, mean=0, stddev=1)
        self.assertEqual(result.shape, shape)

    def test_truncated_normal(self):
        shape = (2, 3)
        result = truncated_normal(shape, mean=0, stddev=1)
        self.assertEqual(result.shape, shape)

    def test_he_normal(self):
        shape = (2, 3)
        result = he_normal(shape)
        self.assertEqual(result.shape, shape)

    def test_he_uniform(self):
        shape = (2, 3)
        result = he_uniform(shape)
        self.assertEqual(result.shape, shape)

    def test_xavier_uniform(self):
        shape = (2, 3)
        result = xavier_uniform(shape)
        self.assertEqual(result.shape, shape)

    def test_xavier_normal(self):
        shape = (2, 3)
        result = xavier_normal(shape)
        self.assertEqual(result.shape, shape)

    def test_matmul(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(3, 2), mstype.float32)
        result = matmul(a, b)
        expected = np.matmul(a.asnumpy(), b.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_add(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(2, 3), mstype.float32)
        result = add(a, b)
        expected = a.asnumpy() + b.asnumpy()
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_minimum(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(2, 3), mstype.float32)
        result = minimum(a, b)
        expected = np.minimum(a.asnumpy(), b.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_reshape(self):
        tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        new_shape = (3, 2)
        reshaped_tensor = reshape(tensor, new_shape)
        expected_output = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(reshaped_tensor.asnumpy(), expected_output)

    def test_concat(self):
        tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
        tensor2 = Tensor(np.array([[5, 6], [7, 8]]))
        concatenated_tensor = concat([tensor1, tensor2], axis=0)
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(concatenated_tensor.asnumpy(), expected_output)

    # def test_convert_to_tensor(self):
    #     value = [1, 2, 3, 4]
    #     tensor = convert_to_tensor(value, dtype=np.float32)
    #     expected_output = np.array([1, 2, 3, 4], dtype=np.float32)
    #     np.testing.assert_array_equal(tensor.asnumpy(), expected_output)

    def test_convert_to_numpy(self):
        tensor = Tensor(np.array([1, 2, 3, 4]))
        numpy_array = convert_to_numpy(tensor)
        expected_output = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(numpy_array, expected_output)

    def test_reduce_mean(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = reduce_mean(a)
        expected = np.mean(a.asnumpy())
        np.testing.assert_almost_equal(result.asnumpy(), expected)

    def test_reduce_max(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = reduce_max(a)
        expected = np.max(a.asnumpy())
        np.testing.assert_almost_equal(result.asnumpy(), expected)

    def test_reduce_min(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = reduce_min(a)
        expected = np.min(a.asnumpy())
        np.testing.assert_almost_equal(result.asnumpy(), expected)

    def test_reshape(self):
        tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        new_shape = (3, 2)
        reshaped_tensor = reshape(tensor, new_shape)
        expected_output = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(reshaped_tensor.asnumpy(), expected_output)

    def test_concat(self):
        tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
        tensor2 = Tensor(np.array([[5, 6], [7, 8]]))
        concatenated_tensor = concat([tensor1, tensor2], axis=0)
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(concatenated_tensor.asnumpy(), expected_output)

    def test_convert_to_tensor(self):
        value = [1, 2, 3, 4]
        tensor = convert_to_tensor(value, dtype=np.float32)
        expected_output = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(tensor.asnumpy(), expected_output)

    def test_convert_to_numpy(self):
        tensor = Tensor(np.array([1, 2, 3, 4]))
        numpy_array = convert_to_numpy(tensor)
        expected_output = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(numpy_array, expected_output)

    def test_sqrt(self):
        tensor = Tensor(np.array([1, 4, 9, 16], dtype=np.float32))
        sqrt_tensor = sqrt(tensor)
        expected_output = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(sqrt_tensor.asnumpy(), expected_output)

    def test_stack(self):
        tensor1 = Tensor(np.array([1, 2]))
        tensor2 = Tensor(np.array([3, 4]))
        stacked_tensor = stack([tensor1, tensor2], axis=0)
        expected_output = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(stacked_tensor.asnumpy(), expected_output)

    def test_meshgrid(self):
        x = Tensor(np.array([1, 2, 3]))
        y = Tensor(np.array([4, 5, 6]))
        grid_x, grid_y = meshgrid(x, y)
        expected_output_x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        expected_output_y = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6]])
        np.testing.assert_array_equal(grid_x.asnumpy(), expected_output_x)
        np.testing.assert_array_equal(grid_y.asnumpy(), expected_output_y)

    def test_arange(self):
        start = 0
        limit = 5
        delta = 1
        arange_tensor = arange(start, limit, delta, dtype=np.int32)
        expected_output = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        np.testing.assert_array_equal(arange_tensor.asnumpy(), expected_output)

    def test_expand_dims(self):
        tensor = Tensor(np.array([1, 2, 3]))
        expanded_tensor = expand_dims(tensor, axis=0)
        expected_output = np.array([[1, 2, 3]])
        np.testing.assert_array_equal(expanded_tensor.asnumpy(), expected_output)

    def test_tile(self):
        tensor = Tensor(np.array([1, 2, 3]))
        tiled_tensor = tile(tensor, multiples=[2])
        expected_output = np.array([1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(tiled_tensor.asnumpy(), expected_output)

    def test_cast(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = cast(a, mstype.int32)
        expected = a.asnumpy().astype(np.int32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_transpose(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = transpose(a, (1, 0))
        expected = np.transpose(a.asnumpy(), (1, 0))
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_gather_nd(self):
        params = Tensor(np.random.rand(2, 3), mstype.float32)
        indices = Tensor([[0, 0], [1, 1]], mstype.int32)
        result = gather_nd(params, indices)
        expected = np.array([params.asnumpy()[0, 0], params.asnumpy()[1, 1]])
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_clip_by_value(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = clip_by_value(a, 0.2, 0.8)
        expected = np.clip(a.asnumpy(), 0.2, 0.8)
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_split(self):
        a = Tensor(np.random.rand(6, 3), mstype.float32)
        result = split(a, 2)
        expected = np.split(a.asnumpy(), 2)
        for res, exp in zip(result, expected):
            np.testing.assert_array_almost_equal(res.asnumpy(), exp)

    def test_floor(self):
        a = Tensor(np.random.rand(2, 3) + 0.5, mstype.float32)
        result = floor(a)
        expected = np.floor(a.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_gather(self):
        params = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
        indices = Tensor(np.array([0, 2]))
        gathered_tensor = gather(params, indices, axis=0)
        expected_output = np.array([[1, 2], [5, 6]])
        np.testing.assert_array_equal(gathered_tensor.asnumpy(), expected_output)

    def test_linspace(self):
        result = linspace(0, 10, 5)
        expected = np.linspace(0, 10, 5)
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_slice(self):
        a = Tensor(np.random.rand(4, 4), mstype.float32)
        result = slice(a, (1, 1), (2, 2))
        expected = a.asnumpy()[1:3, 1:3]
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_add_n(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(2, 3), mstype.float32)
        c = Tensor(np.random.rand(2, 3), mstype.float32)
        result = add_n([a, b, c])
        expected = a.asnumpy() + b.asnumpy() + c.asnumpy()
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_one_hot(self):
        indices = Tensor([0, 1, 2], mstype.int32)
        depth = 3
        result = OneHot(depth=depth)(indices)
        expected = np.eye(depth)[indices.asnumpy()]
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_embedding_lookup(self):
        params = Tensor(np.random.rand(10, 5), ms.float32)
        ids = Tensor([1, 2, 3], ms.int32)
        embedding_lookup = EmbeddingLookup()
        output = embedding_lookup.construct(params, ids)
        self.assertEqual(output.shape, (3, 5))

    def test_count_nonzero(self):
        x = Tensor([0, 1, 2, 0, 3], ms.int32)
        count_nonzero = CountNonzero()
        output = count_nonzero(x)
        self.assertEqual(output, 3)

    def test_resize(self):
        inputs = Tensor(np.random.rand(1, 3, 3, 1), ms.float32)
        resize = Resize(scale=(2, 2), method="nearest")
        output = resize.construct(inputs)
        self.assertEqual(output.shape, (1, 6, 6, 1))

    def test_zero_padding_1d(self):
        inputs = Tensor(np.random.rand(1, 3, 3), ms.float32)
        zero_padding_1d = ZeroPadding1D(padding=(1, 1), data_format="channels_last")
        output = zero_padding_1d.construct(inputs)
        self.assertEqual(output.shape, (1, 5, 3))

    def test_zero_padding_2d(self):
        inputs = Tensor(np.random.rand(1, 3, 3, 1), ms.float32)
        zero_padding_2d = ZeroPadding2D(padding=((1, 1), (1, 1)), data_format="channels_last")
        output = zero_padding_2d.construct(inputs)
        self.assertEqual(output.shape, (1, 5, 5, 1))

    def test_zero_padding_3d(self):
        inputs = Tensor(np.random.rand(1, 3, 3, 3, 1), ms.float32)
        zero_padding_3d = ZeroPadding3D(padding=((1, 1), (1, 1), (1, 1)), data_format="channels_last")
        output = zero_padding_3d.construct(inputs)
        self.assertEqual(output.shape, (1, 5, 5, 5, 1))

    def test_ceil(self):
        x = Tensor([1.1, 2.5, 3.7], ms.float32)
        result = ceil(x)
        expected = Tensor([2.0, 3.0, 4.0], ms.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())

    def test_multiply(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([4, 5, 6], ms.float32)
        result = multiply(x, y)
        expected = Tensor([4, 10, 18], ms.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())

    def test_divide(self):
        x = Tensor([4, 9, 16], ms.float32)
        y = Tensor([2, 3, 4], ms.float32)
        result = divide(x, y)
        expected = Tensor([2, 3, 4], ms.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())

    def test_identity(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = identity(x)
        np.testing.assert_array_equal(result.asnumpy(), x.asnumpy())

    # def test_batch_to_space(self):
    #     x = Tensor(np.random.rand(4, 1, 1, 1), ms.float32)
    #     block_size = 2
    #     crops = [[0, 0], [0, 0]]
    #     bts = BatchToSpace(block_size, crops)
    #     result = bts(x)
    #     expected_shape = (1, 2, 2, 1)
    #     self.assertEqual(result.shape, expected_shape)

    def test_depth_to_space(self):
        x = Tensor(np.random.rand(1, 4, 1, 1), ms.float32)
        block_size = 2
        dts = DepthToSpace(block_size, data_format="NCHW")
        result = dts(x)
        expected_shape = (1, 1, 2, 2)
        self.assertEqual(result.shape, expected_shape)

    def test_triu(self):
        x = Tensor(np.random.rand(3, 3), ms.float32)
        result = triu(x)
        expected = np.triu(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_tril(self):
        x = Tensor(np.random.rand(3, 3), ms.float32)
        result = tril(x)
        expected = np.tril(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_abs(self):
        x = Tensor([-1, -2, 3], ms.float32)
        result = abs(x)
        expected = Tensor([1, 2, 3], ms.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())

    def test_acos(self):
        x = Tensor([1, 0, -1], ms.float32)
        result = acos(x)
        expected = np.arccos(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_acosh(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = acosh(x)
        expected = np.arccosh(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_angle(self):
        x = Tensor([1 + 1j, 1 - 1j], ms.complex64)
        result = angle(x)
        expected = np.angle(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_argmax(self):
        x = Tensor([1, 3, 2], ms.float32)
        result = argmax(x)
        expected = 1
        self.assertEqual(result.asnumpy(), expected)

    def test_argmin(self):
        x = Tensor([1, 3, 2], ms.float32)
        result = argmin(x)
        expected = 0
        self.assertEqual(result.asnumpy(), expected)

    def test_asin(self):
        x = Tensor([0, 0.5, 1], ms.float32)
        result = asin(x)
        expected = np.arcsin(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_asinh(self):
        x = Tensor([0, 0.5, 1], ms.float32)
        result = asinh(x)
        expected = np.arcsinh(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_atan(self):
        x = Tensor([0, 1, -1], ms.float32)
        result = atan(x)
        expected = np.arctan(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_atanh(self):
        x = Tensor([0, 0.5, -0.5], ms.float32)
        result = atanh(x)
        expected = np.arctanh(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_cos(self):
        x = Tensor([0, np.pi / 2, np.pi], ms.float32)
        result = cos(x)
        expected = np.cos(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_cosh(self):
        x = Tensor([0, 1, -1], ms.float32)
        result = cosh(x)
        expected = np.cosh(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_count_nonzero(self):
        x = Tensor([0, 1, 2, 0, 3], ms.float32)
        result = count_nonzero(x)
        expected = 3
        self.assertEqual(result.asnumpy(), expected)

    def test_cumprod(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = cumprod(x)
        expected = np.cumprod(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_cumsum(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = cumsum(x)
        expected = np.cumsum(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_equal(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 2, 4], ms.float32)
        result = equal(x, y)
        expected = np.equal(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_exp(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = exp(x)
        expected = np.exp(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_floordiv(self):
        x = Tensor([4, 9, 16], ms.float32)
        y = Tensor([2, 3, 4], ms.float32)
        result = floordiv(x, y)
        expected = np.floor_divide(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_floormod(self):
        x = Tensor([4, 9, 16], ms.float32)
        y = Tensor([2, 3, 4], ms.float32)
        result = floormod(x, y)
        expected = np.mod(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_greater(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 1, 4], ms.float32)
        result = greater(x, y)
        expected = np.greater(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_greater_equal(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 1, 4], ms.float32)
        result = greater_equal(x, y)
        expected = np.greater_equal(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_is_inf(self):
        x = Tensor([1, 2, np.inf], ms.float32)
        result = is_inf(x)
        expected = np.isinf(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_is_nan(self):
        x = Tensor([1, 2, np.nan], ms.float32)
        result = is_nan(x)
        expected = np.isnan(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_l2_normalize(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = l2_normalize(x)
        expected = x.asnumpy() / np.linalg.norm(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_less(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 1, 4], ms.float32)
        result = less(x, y)
        expected = np.less(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_less_equal(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 1, 4], ms.float32)
        result = less_equal(x, y)
        expected = np.less_equal(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_log(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = log(x)
        expected = np.log(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_log_sigmoid(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-x.asnumpy())))
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_maximum(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 3, 2], ms.float32)
        result = maximum(x, y)
        expected = np.maximum(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_negative(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = negative(x)
        expected = np.negative(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_not_equal(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 3, 2], ms.float32)
        result = not_equal(x, y)
        expected = np.not_equal(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_pow(self):
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([1, 3, 2], ms.float32)
        result = pow(x, y)
        expected = np.power(x.asnumpy(), y.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_real(self):
        x = Tensor([1 + 1j, 1 - 1j], ms.complex64)
        result = real(x)
        expected = np.real(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_reciprocal(self):
        x = Tensor([1, 2, 4], ms.float32)
        result = reciprocal(x)
        expected = np.reciprocal(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_reduce_prod(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = reduce_prod(x)
        expected = np.prod(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_reduce_std(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = reduce_std(x)
        expected = np.std(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_reduce_sum(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = reduce_sum(x)
        expected = np.sum(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_reduce_variance(self):
        x = Tensor([1, 2, 3], ms.float32)
        result = reduce_variance(x)
        expected = np.var(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_round(self):
        x = Tensor([1.1, 2.5, 3.7], ms.float32)
        result = round(x)
        expected = np.round(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_rsqrt(self):
        x = Tensor([1, 4, 9], ms.float32)
        result = rsqrt(x)
        expected = 1 / np.sqrt(x.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_segment_max(self):
        x = Tensor([1, 2, 3, 4], ms.float32)
        segment_ids = Tensor([0, 0, 1, 1], ms.int32)
        result = segment_max(x, segment_ids)
        expected = np.array([2, 4], dtype=np.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_segment_mean(self):
        x = Tensor([1, 2, 3, 4], ms.float32)
        segment_ids = Tensor([0, 0, 1, 1], ms.int32)
        result = segment_mean(x, segment_ids)
        expected = np.array([1.5, 3.5], dtype=np.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_segment_min(self):
        x = Tensor([1, 2, 3, 4], ms.float32)
        segment_ids = Tensor([0, 0, 1, 1], ms.int32)
        result = segment_min(x, segment_ids)
        expected = np.array([1, 3], dtype=np.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_segment_prod(self):
        x = Tensor([1, 2, 3, 4], ms.float32)
        segment_ids = Tensor([0, 0, 1, 1], ms.int32)
        result = segment_prod(x, segment_ids)
        expected = np.array([2, 12], dtype=np.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_segment_sum(self):
        x = Tensor([1, 2, 3, 4], ms.float32)
        segment_ids = Tensor([0, 0, 1, 1], ms.int32)
        result = segment_sum(x, segment_ids)
        expected = np.array([3, 7], dtype=np.float32)
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_sigmoid(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = sigmoid(a)
        expected = 1 / (1 + np.exp(-a.asnumpy()))
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_sign(self):
        x = ms.Tensor(np.array([-1.0, 0.0, 1.0]), ms.float32)
        expected = np.sign(np.array([-1.0, 0.0, 1.0]))
        output = sign(x).asnumpy()
        np.testing.assert_array_equal(output, expected)

    def test_sin(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = sin(a)
        expected = np.sin(a.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_sinh(self):
        x = ms.Tensor(np.array([-1.0, 0.0, 1.0]), ms.float32)
        expected = np.sinh(np.array([-1.0, 0.0, 1.0]))
        output = sinh(x).asnumpy()
        np.testing.assert_almost_equal(output, expected, decimal=5)

    def test_softplus(self):
        x = ms.Tensor(np.array([-1.0, 0.0, 1.0]), ms.float32)
        expected = np.log(np.exp(np.array([-1.0, 0.0, 1.0])) + 1)
        output = softplus(x).asnumpy()
        np.testing.assert_almost_equal(output, expected, decimal=5)

    def test_square(self):
        x = ms.Tensor(np.array([-1.0, 0.0, 1.0]), ms.float32)
        expected = np.square(np.array([-1.0, 0.0, 1.0]))
        output = square(x).asnumpy()
        np.testing.assert_array_equal(output, expected)

    def test_squared_difference(self):
        x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
        y = ms.Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
        expected = np.square(np.array([1.0, 2.0, 3.0]) - np.array([4.0, 5.0, 6.0]))
        output = squared_difference(x, y).asnumpy()
        np.testing.assert_array_equal(output, expected)

    def test_subtract(self):
        x = ms.Tensor(np.array([4.0, 9.0, 16.0]), ms.float32)
        y = ms.Tensor(np.array([2.0, 3.0, 4.0]), ms.float32)
        expected = np.array([2.0, 6.0, 12.0])
        output = subtract(x, y).asnumpy()
        np.testing.assert_array_equal(output, expected)

    def test_tan(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = tan(a)
        expected = np.tan(a.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_tanh(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        expected = np.tanh(x.asnumpy())
        output = tanh(x)
        np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=6)

    def test_any(self):
        x = Tensor(np.array([[False, False], [True, False]], dtype=np.bool_))
        expected = np.any(x.asnumpy())
        output = any(x)
        self.assertEqual(output.asnumpy(), expected)

        expected_axis_0 = np.any(x.asnumpy(), axis=0)
        output_axis_0 = any(x, axis=0)
        np.testing.assert_array_equal(output_axis_0.asnumpy(), expected_axis_0)

        expected_axis_1 = np.any(x.asnumpy(), axis=1)
        output_axis_1 = any(x, axis=1)
        np.testing.assert_array_equal(output_axis_1.asnumpy(), expected_axis_1)

    def test_all(self):
        x = Tensor(np.array([[True, False], [True, True]], dtype=np.bool_))
        expected = np.all(x.asnumpy())
        output = all(x)
        self.assertEqual(output.asnumpy(), expected)

        expected_axis_0 = np.all(x.asnumpy(), axis=0)
        output_axis_0 = all(x, axis=0)
        np.testing.assert_array_equal(output_axis_0.asnumpy(), expected_axis_0)

        expected_axis_1 = np.all(x.asnumpy(), axis=1)
        output_axis_1 = all(x, axis=1)
        np.testing.assert_array_equal(output_axis_1.asnumpy(), expected_axis_1)

    # def test_log(self):
    #     a = Tensor(np.random.rand(2, 3) + 1, mstype.float32)
    #     result = log(a)
    #     expected = np.log(a.asnumpy())
    #     np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    # def test_exp(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = exp(a)
    #     expected = np.exp(a.asnumpy())
    #     np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    # def test_sqrt(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = sqrt(a)
    #     expected = np.sqrt(a.asnumpy())
    #     np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    # def test_pow(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = pow(a, b)
    #     expected = np.power(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    # def test_abs(self):
    #     a = Tensor(np.random.rand(2, 3) - 0.5, mstype.float32)
    #     result = abs(a)
    #     expected = np.abs(a.asnumpy())
    #     np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    # def test_argmax(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = argmax(a)
    #     expected = np.argmax(a.asnumpy())
    #     self.assertEqual(result.asnumpy(), expected)

    # def test_argmin(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = argmin(a)
    #     expected = np.argmin(a.asnumpy())
    #     self.assertEqual(result.asnumpy(), expected)

    # def test_equal(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = equal(a, b)
    #     expected = np.equal(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    # def test_not_equal(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = not_equal(a, b)
    #     expected = np.not_equal(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    # def test_greater(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = greater(a, b)
    #     expected = np.greater(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    # def test_greater_equal(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = greater_equal(a, b)
    #     expected = np.greater_equal(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    # def test_less(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = less(a, b)
    #     expected = np.less(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    # def test_less_equal(self):
    #     a = Tensor(np.random.rand(2, 3), mstype.float32)
    #     b = Tensor(np.random.rand(2, 3), mstype.float32)
    #     result = less_equal(a, b)
    #     expected = np.less_equal(a.asnumpy(), b.asnumpy())
    #     np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_logical_and(self):
        a = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        b = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        result = logical_and(a, b)
        expected = np.logical_and(a.asnumpy(), b.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_logical_or(self):
        a = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        b = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        result = logical_or(a, b)
        expected = np.logical_or(a.asnumpy(), b.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_logical_not(self):
        a = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        result = logical_not(a)
        expected = np.logical_not(a.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_logical_xor(self):
        a = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        b = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        result = logical_xor(a, b)
        expected = np.logical_xor(a.asnumpy(), b.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_argsort(self):
        x = Tensor(np.array([[3, 1, 2], [9, 7, 8]], dtype=np.float32))
        expected = np.argsort(x.asnumpy(), axis=-1)
        output = argsort(x, axis=-1)
        np.testing.assert_array_equal(output.asnumpy(), expected)

        expected_descending = np.argsort(-x.asnumpy(), axis=-1)
        output_descending = argsort(x, axis=-1, descending=True)
        np.testing.assert_array_equal(output_descending.asnumpy(), expected_descending)

    def test_bmm(self):
        x = Tensor(np.random.rand(10, 3, 4).astype(np.float32))
        y = Tensor(np.random.rand(10, 4, 5).astype(np.float32))
        expected = np.matmul(x.asnumpy(), y.asnumpy())
        output = bmm(x, y)
        np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=6)

    def test_where(self):
        condition = Tensor(np.random.rand(2, 3) > 0.5, mstype.bool_)
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(2, 3), mstype.float32)
        result = where(condition, a, b)
        expected = np.where(condition.asnumpy(), a.asnumpy(), b.asnumpy())
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_ones_like(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = ones_like(a)
        expected = np.ones_like(a.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_zeros_like(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        result = zeros_like(a)
        expected = np.zeros_like(a.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_squeeze(self):
        a = Tensor(np.random.rand(1, 2, 3, 1), mstype.float32)
        result = squeeze(a)
        expected = np.squeeze(a.asnumpy())
        np.testing.assert_array_equal(result.asnumpy(), expected)

    def test_unsorted_segment_sum(self):
        a = Tensor(np.random.rand(6, 3), mstype.float32)
        segment_ids = Tensor([0, 1, 2, 0, 1, 2], mstype.int32)
        num_segments = 3
        result = unsorted_segment_sum(a, segment_ids, num_segments)
        expected = np.array([a.asnumpy()[[0, 3]].sum(axis=0), a.asnumpy()[[1, 4]].sum(axis=0), a.asnumpy()[[2, 5]].sum(axis=0)])
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_unsorted_segment_mean(self):
        a = Tensor(np.random.rand(6, 3), mstype.float32)
        segment_ids = Tensor([0, 1, 2, 0, 1, 2], mstype.int32)
        num_segments = 3
        result = unsorted_segment_mean(a, segment_ids, num_segments)
        expected = np.array([a.asnumpy()[[0, 3]].mean(axis=0), a.asnumpy()[[1, 4]].mean(axis=0), a.asnumpy()[[2, 5]].mean(axis=0)])
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_unsorted_segment_min(self):
        a = Tensor(np.random.rand(6, 3), mstype.float32)
        segment_ids = Tensor([0, 1, 2, 0, 1, 2], mstype.int32)
        num_segments = 3
        result = unsorted_segment_min(a, segment_ids, num_segments)
        expected = np.array([a.asnumpy()[[0, 3]].min(axis=0), a.asnumpy()[[1, 4]].min(axis=0), a.asnumpy()[[2, 5]].min(axis=0)])
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_unsorted_segment_max(self):
        a = Tensor(np.random.rand(6, 3), mstype.float32)
        segment_ids = Tensor([0, 1, 2, 0, 1, 2], mstype.int32)
        num_segments = 3
        result = unsorted_segment_max(a, segment_ids, num_segments)
        expected = np.array([a.asnumpy()[[0, 3]].max(axis=0), a.asnumpy()[[1, 4]].max(axis=0), a.asnumpy()[[2, 5]].max(axis=0)])
        np.testing.assert_array_almost_equal(result.asnumpy(), expected)

    def test_set_seed(self):
        set_seed(42)
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        b = Tensor(np.random.rand(2, 3), mstype.float32)
        self.assertNotEqual(a.asnumpy().tolist(), b.asnumpy().tolist())

    def test_is_tensor(self):
        a = Tensor(np.random.rand(2, 3), mstype.float32)
        self.assertTrue(is_tensor(a))
        self.assertFalse(is_tensor(np.random.rand(2, 3)))

    def test_tensor_scatter_nd_update(self):
        tensor = Tensor(np.array([[1, 2], [3, 4]]))
        indices = Tensor(np.array([[0, 0], [1, 1]]))
        updates = Tensor(np.array([5, 6]))
        result = tensor_scatter_nd_update(tensor, indices, updates)
        expected = Tensor(np.array([[5, 2], [3, 6]]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_diag(self):
        input = Tensor(np.array([1, 2, 3]))
        result = diag(input)
        expected = Tensor(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_mask_select(self):
        x = Tensor(np.array([1, 2, 3, 4]))
        mask = Tensor(np.array([True, False, True, False]))
        result = mask_select(x, mask)
        expected = Tensor(np.array([1, 3]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_eye(self):
        result = eye(3)
        expected = Tensor(np.eye(3))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_einsum(self):
        x = Tensor(np.array([[1, 2], [3, 4]]))
        y = Tensor(np.array([[1, 0], [0, 1]]))
        result = einsum("ij,jk->ik", x, y)
        expected = Tensor(np.array([[1, 2], [3, 4]]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_roll(self):
        x = Tensor(np.array([1, 2, 3, 4]))
        result = roll(x, shifts=2)
        expected = Tensor(np.array([3, 4, 1, 2]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))

    def test_logsoftmax(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        result = logsoftmax(x)
        expected = Tensor(np.log(np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))))
        self.assertTrue(np.allclose(result.asnumpy(), expected.asnumpy()))

    def test_topk(self):
        x = Tensor(np.array([1, 3, 2, 4]))
        values, indices = topk(x, k=2)
        expected_values = Tensor(np.array([4, 3]))
        expected_indices = Tensor(np.array([3, 1]))
        self.assertTrue(np.array_equal(values.asnumpy(), expected_values.asnumpy()))
        self.assertTrue(np.array_equal(indices.asnumpy(), expected_indices.asnumpy()))

    def test_numel(self):
        x = Tensor(np.array([[1, 2], [3, 4]]))
        result = numel(x)
        expected = 4
        self.assertEqual(result, expected)

    # def test_histogram(self):
    #     with self.assertRaises(NotImplementedError):
    #         histogram(Tensor(np.array([1, 2, 3])))

    # def test_flatten(self):
    #     with self.assertRaises(NotImplementedError):
    #         flatten(Tensor(np.array([[1, 2], [3, 4]])))

    # def test_interpolate(self):
    #     with self.assertRaises(NotImplementedError):
    #         interpolate(Tensor(np.array([[1, 2], [3, 4]])))

    # def test_index_select(self):
    #     with self.assertRaises(NotImplementedError):
    #         index_select(Tensor(np.array([1, 2, 3])), Tensor(np.array([0, 2])))

    # def test_dot(self):
    #     with self.assertRaises(NotImplementedError):
    #         dot(Tensor(np.array([1, 2, 3])), Tensor(np.array([4, 5, 6])))

    # def test_swish(self):
    #     swish = Swish()
    #     with self.assertRaises(NotImplementedError):
    #         swish(Tensor(np.array([1, 2, 3])))

    # def test_expand(self):
    #     with self.assertRaises(NotImplementedError):
    #         expand(Tensor(np.array([1, 2, 3])), (3, 3))

    # def test_unique(self):
    #     with self.assertRaises(NotImplementedError):
    #         unique(Tensor(np.array([1, 2, 2, 3])))

    # def test_flip(self):
    #     with self.assertRaises(NotImplementedError):
    #         flip(Tensor(np.array([1, 2, 3])), axis=0)

    def test_mv(self):
        x = Tensor(np.array([[1, 2], [3, 4]]))
        vec = Tensor(np.array([1, 2]))
        result = mv(x, vec)
        expected = Tensor(np.array([5, 11]))
        self.assertTrue(np.array_equal(result.asnumpy(), expected.asnumpy()))


if __name__ == "__main__":
    unittest.main()
