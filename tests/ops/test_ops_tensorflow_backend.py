import unittest

import tensorflow as tf
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TL_BACKEND"] = "tensorflow"


from tensorlayerx.backend.ops.tensorflow_backend import *
from tests.utils import CustomTestCase


class TestTensorFlowBackend(CustomTestCase):

    def test_dtype_str(self):
        self.assertEqual(dtype_str("float32"), tf.float32)
        self.assertEqual(dtype_str("int64"), tf.int64)
        with self.assertRaises(NotImplementedError):
            dtype_str("unknown")

    def test_get_tensor_shape(self):
        x = tf.zeros((32, 3, 3, 32))
        self.assertEqual(get_tensor_shape(x), [32, 3, 3, 32])

    def test_zeros(self):
        x = zeros((32, 3, 3, 32), dtype="int32")
        self.assertTrue(tf.reduce_all(tf.equal(x, 0)))

    def test_ones(self):
        x = ones((32, 3, 3, 32), dtype="int32")
        self.assertTrue(tf.reduce_all(tf.equal(x, 1)))

    def test_constant(self):
        x = constant(0.5, (32, 3, 3, 32), dtype="float32")
        self.assertTrue(tf.reduce_all(tf.equal(x, 0.5)))

    def test_random_uniform(self):
        x = random_uniform((32, 3, 3, 32), minval=0, maxval=1, dtype="float32")
        self.assertTrue(tf.reduce_all(tf.greater_equal(x, 0)))
        self.assertTrue(tf.reduce_all(tf.less(x, 1)))

    def test_random_normal(self):
        x = random_normal((32, 3, 3, 32), mean=0.0, stddev=1.0, dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    def test_truncated_normal(self):
        x = truncated_normal((32, 3, 3, 32), mean=0.0, stddev=1.0, dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    def test_he_normal(self):
        x = he_normal((32, 3, 3, 32), dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    def test_he_uniform(self):
        x = he_uniform((32, 3, 3, 32), dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    def test_xavier_normal(self):
        x = xavier_normal((32, 3, 3, 32), dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    def test_xavier_uniform(self):
        x = xavier_uniform((32, 3, 3, 32), dtype="float32")
        self.assertEqual(x.shape, (32, 3, 3, 32))

    # def test_Variable(self):
    #     x = Variable(tf.ones((10, 20)), name="w")
    #     self.assertEqual(x.shape, (10, 20))

    def test_matmul(self):
        a = tf.random.normal((2, 3))
        b = tf.random.normal((3, 2))
        c = matmul(a, b)
        self.assertEqual(c.shape, (2, 2))

    def test_add(self):
        a = tf.ones((10, 20))
        b = tf.ones((20,))
        c = add(a, b)
        self.assertEqual(c.shape, (10, 20))

    def test_minimum(self):
        a = tf.constant([0.0, 0.0, 0.0, 0.0])
        b = tf.constant([-5.0, -2.0, 0.0, 3.0])
        c = minimum(a, b)
        self.assertTrue(tf.reduce_all(tf.equal(c, [-5.0, -2.0, 0.0, 0.0])))

    def test_reshape(self):
        a = tf.constant([0.0, 1.0, 2.0, 3.0])
        b = reshape(a, [2, 2])
        self.assertEqual(b.shape, (2, 2))

    def test_concat(self):
        a = tf.constant([0.0, 0.0, 0.0, 0.0])
        b = tf.constant([-5.0, -2.0, 0.0, 3.0])
        c = concat([a, b], 0)
        self.assertEqual(c.shape, (8,))

    def test_convert_to_tensor(self):
        a = np.ones((10, 10))
        b = convert_to_tensor(a)
        self.assertEqual(b.shape, (10, 10))

    def test_convert_to_numpy(self):
        a = tf.ones((10, 10))
        b = convert_to_numpy(a)
        self.assertEqual(b.shape, (10, 10))

    def test_sqrt(self):
        a = tf.constant([0.0, 1.0, 4.0], dtype=tf.float32)
        b = sqrt(a)
        self.assertTrue(tf.reduce_all(tf.equal(b, [0.0, 1.0, 2.0])))

    def test_reduce_mean(self):
        a = tf.random.normal((3, 4))
        b = reduce_mean(a, axis=1, keepdims=False)
        self.assertEqual(b.shape, (3,))

    def test_reduce_max(self):
        a = tf.random.normal((3, 4))
        b = reduce_max(a, axis=1, keepdims=False)
        self.assertEqual(b.shape, (3,))

    def test_reduce_min(self):
        a = tf.random.normal((3, 4))
        b = reduce_min(a, axis=1, keepdims=False)
        self.assertEqual(b.shape, (3,))

    # def test_pad(self):
    #     a = tf.constant([[1, 2, 3], [4, 5, 6]])
    #     paddings = [[1, 1], [2, 2]]
    #     b = pad(a, paddings)
    #     self.assertEqual(b.shape, (4, 7))

    def test_stack(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([1, 2, 3])
        c = stack([a, b])
        self.assertEqual(c.shape, (2, 3))

    def test_arange(self):
        a = arange(0, 10, 1)
        self.assertEqual(a.shape, (10,))

    def test_expand_dims(self):
        a = tf.ones([1, 2, 3])
        b = expand_dims(a, axis=0)
        self.assertEqual(b.shape, (1, 1, 2, 3))

    def test_tile(self):
        a = tf.constant([[1, 2, 3], [1, 2, 3]])
        b = tile(a, [2, 1])
        self.assertEqual(b.shape, (4, 3))

    def test_cast(self):
        a = tf.constant([1.0, 2.0, 3.0])
        b = cast(a, tf.int32)
        self.assertEqual(b.dtype, tf.int32)

    def test_transpose(self):
        a = tf.constant([[1, 2, 3], [4, 5, 6]])
        b = transpose(a, perm=[1, 0])
        self.assertEqual(b.shape, (3, 2))

    def test_gather_nd(self):
        a = tf.constant([[1, 2], [3, 4]])
        b = gather_nd(a, [[0, 0], [1, 1]])
        self.assertTrue(tf.reduce_all(tf.equal(b, [1, 4])))

    def test_clip_by_value(self):
        a = tf.constant([1.0, 2.0, 3.0, 4.0])
        b = clip_by_value(a, 1.5, 3.5)
        self.assertTrue(tf.reduce_all(tf.equal(b, [1.5, 2.0, 3.0, 3.5])))

    def test_split(self):
        a = tf.ones([3, 9, 5])
        b = split(a, 3, axis=1)
        self.assertEqual(len(b), 3)
        self.assertEqual(b[0].shape, (3, 3, 5))

    def test_floor(self):
        a = tf.constant([1.23, 2.56, 3.589])
        b = floor(a)
        self.assertTrue(tf.reduce_all(tf.equal(b, [1.0, 2.0, 3.0])))

    def test_gather(self):
        a = tf.constant([[0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
        b = gather(a, indices=[3, 1])
        self.assertEqual(b.shape, (2, 3))

    def test_linspace(self):
        a = linspace(0.0, 1.0, 10)
        self.assertEqual(a.shape, (10,))

    def test_slice(self):
        a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = slice(a, [1, 0], [1, 2])
        self.assertTrue(tf.reduce_all(tf.equal(b, [[4, 5]])))

    def test_add_n(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = add_n([a, b])
        self.assertTrue(tf.reduce_all(tf.equal(c, [5, 7, 9])))

    def test_one_hot(self):
        a = tf.constant([0, 1, 2])
        b = OneHot(depth=3)(a)
        self.assertEqual(b.shape, (3, 3))

    def test_l2_normalize(self):
        a = tf.constant([1.0, 2.0, 3.0])
        b = L2Normalize()(a)
        self.assertTrue(np.allclose(b.numpy(), [0.26726124, 0.5345225, 0.8017837]))

    def test_embedding_lookup(self):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = tf.constant([0, 2])
        c = EmbeddingLookup()(a, b)
        self.assertTrue(tf.reduce_all(tf.equal(c, [[1.0, 2.0], [5.0, 6.0]])))

    def test_nce_loss(self):
        weights = tf.random.normal([10000, 128])
        biases = tf.random.normal([10000])
        labels = tf.random.uniform([64, 1], maxval=10000, dtype=tf.int64)
        inputs = tf.random.normal([64, 128])
        loss = NCELoss()(weights, biases, labels, inputs, num_sampled=64, num_classes=10000)
        self.assertEqual(loss.shape, (64,))

    def test_not_equal(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([1, 3, 5])
        c = NotEqual()(a, b)
        self.assertTrue(tf.reduce_all(tf.equal(c, [False, True, True])))

    def test_count_nonzero(self):
        a = tf.constant([0, 1, 2, 0, 3])
        b = CountNonzero()(a)
        self.assertEqual(b.numpy(), 3)

    def test_resize(self):
        a = tf.random.normal([1, 32, 32, 3])
        b = Resize(scale=[2, 2], method="bilinear")(a)
        self.assertEqual(b.shape, (1, 64, 64, 3))

    # def test_zero_padding_1d(self):
    #     a = tf.random.normal([1, 10, 3])
    #     b = ZeroPadding1D(padding=(1, 1), data_format="channels_last")(a)
    #     self.assertEqual(b.shape, (1, 12, 3))

    # def test_zero_padding_2d(self):
    #     a = tf.random.normal([1, 10, 10, 3])
    #     b = ZeroPadding2D(padding=((1, 1), (1, 1)), data_format="channels_last")(a)
    #     self.assertEqual(b.shape, (1, 12, 12, 3))

    # def test_zero_padding_3d(self):
    #     a = tf.random.normal([1, 10, 10, 10, 3])
    #     b = ZeroPadding3D(padding=((1, 1), (1, 1), (1, 1)), data_format="channels_last")(a)
    #     self.assertEqual(b.shape, (1, 12, 12, 12, 3))

    def test_ceil(self):
        a = tf.constant([0.9142202, 0.72091234])
        b = ceil(a)
        self.assertTrue(tf.reduce_all(tf.equal(b, [1.0, 1.0])))

    def test_multiply(self):
        a = tf.constant([0.9142202, 0.72091234])
        b = multiply(a, a)
        self.assertTrue(np.allclose(b.numpy(), [0.835798, 0.519717]))

    def test_divide(self):
        a = tf.constant([0.9142202, 0.72091234])
        b = divide(a, a)
        self.assertTrue(tf.reduce_all(tf.equal(b, [1.0, 1.0])))

    def test_identity(self):
        a = tf.constant([1.0, 2.0, 3.0])
        b = identity(a)
        self.assertTrue(tf.reduce_all(tf.equal(a, b)))

    def test_triu(self):
        a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = triu(a, diagonal=1)
        self.assertTrue(tf.reduce_all(tf.equal(b, [[0, 2, 3], [0, 0, 6], [0, 0, 0]])))

    def test_tril(self):
        a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = tril(a, diagonal=-1)
        self.assertTrue(tf.reduce_all(tf.equal(b, [[0, 0, 0], [4, 0, 0], [7, 8, 0]])))

    def test_abs(self):
        a = tf.constant([-1.0, 0.0, 1.0])
        b = abs(a)
        self.assertTrue(tf.reduce_all(tf.equal(b, [1.0, 0.0, 1.0])))

    def test_acos(self):
        a = tf.constant([1.0, 0.0, -1.0])
        b = acos(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 1.5707964, 3.1415927]))

    def test_acosh(self):
        a = tf.constant([1.0, 2.0, 3.0])
        b = acosh(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 1.3169579, 1.7627472]))

    def test_angle(self):
        a = tf.constant([2.15 + 3.57j, 3.89 + 6.54j])
        b = angle(a)
        self.assertTrue(np.allclose(b.numpy(), [1.0277026, 1.033215]))

    def test_argmax(self):
        a = tf.constant([10, 20, 5, 6, 15])
        b = argmax(a)
        self.assertEqual(b.numpy(), 1)

    def test_argmin(self):
        a = tf.constant([10, 20, 5, 6, 15])
        b = argmin(a)
        self.assertEqual(b.numpy(), 2)

    def test_asin(self):
        a = tf.constant([0.0, 0.5, 1.0])
        b = asin(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 0.5235988, 1.5707964]))

    def test_asinh(self):
        a = tf.constant([0.0, 0.5, 1.0])
        b = asinh(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 0.4812118, 0.8813736]))

    def test_atan(self):
        a = tf.constant([0.0, 1.0, -1.0])
        b = atan(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 0.7853982, -0.7853982]))

    def test_atanh(self):
        a = tf.constant([0.0, 0.5, -0.5])
        b = atanh(a)
        self.assertTrue(np.allclose(b.numpy(), [0.0, 0.5493061, -0.5493061]))

    def test_cos(self):
        x = tf.constant([0.0, np.pi / 2], dtype=tf.float32)
        y = cos(x)
        self.assertTrue(np.allclose(y.numpy(), np.cos([0.0, np.pi / 2])))

    def test_cosh(self):
        x = tf.constant([0.0, 1.0], dtype=tf.float32)
        y = cosh(x)
        self.assertTrue(np.allclose(y.numpy(), np.cosh([0.0, 1.0])))

    def test_count_nonzero(self):
        x = tf.constant([0, 1, 2, 0, 3], dtype=tf.int32)
        y = count_nonzero(x)
        self.assertEqual(y.numpy(), 3)

    def test_cumprod(self):
        x = tf.constant([1, 2, 3], dtype=tf.float32)
        y = cumprod(x)
        self.assertTrue(np.allclose(y.numpy(), np.cumprod([1, 2, 3])))

    def test_cumsum(self):
        x = tf.constant([1, 2, 3], dtype=tf.float32)
        y = cumsum(x)
        self.assertTrue(np.allclose(y.numpy(), np.cumsum([1, 2, 3])))

    def test_equal(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([1, 2, 4], dtype=tf.int32)
        z = equal(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [True, True, False]))

    def test_exp(self):
        x = tf.constant([1, 2, 3], dtype=tf.float32)
        y = exp(x)
        self.assertTrue(np.allclose(y.numpy(), np.exp([1, 2, 3])))

    def test_floordiv(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = floordiv(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [0, 1, 1]))

    def test_floormod(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = floormod(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [1, 0, 1]))

    def test_greater(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = greater(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [False, False, True]))

    def test_greater_equal(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = greater_equal(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [False, True, True]))

    def test_is_inf(self):
        x = tf.constant([1.0, 2.0, np.inf], dtype=tf.float32)
        y = is_inf(x)
        self.assertTrue(np.array_equal(y.numpy(), [False, False, True]))

    def test_is_nan(self):
        x = tf.constant([1.0, 2.0, np.nan], dtype=tf.float32)
        y = is_nan(x)
        self.assertTrue(np.array_equal(y.numpy(), [False, False, True]))

    def test_l2_normalize(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = l2_normalize(x)
        self.assertTrue(np.allclose(y.numpy(), x.numpy() / np.linalg.norm(x.numpy())))

    def test_less(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = less(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [True, False, False]))

    def test_less_equal(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 2, 2], dtype=tf.int32)
        z = less_equal(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [True, True, False]))

    def test_log(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = log(x)
        self.assertTrue(np.allclose(y.numpy(), np.log([1.0, 2.0, 3.0])))

    def test_log_sigmoid(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = log_sigmoid(x)
        self.assertTrue(np.allclose(y.numpy(), -np.logaddexp(0, -x.numpy())))

    def test_maximum(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([2, 1, 4], dtype=tf.int32)
        z = maximum(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [2, 2, 4]))

    def test_negative(self):
        x = tf.constant([1, -2, 3], dtype=tf.int32)
        y = negative(x)
        self.assertTrue(np.array_equal(y.numpy(), [-1, 2, -3]))

    def test_not_equal(self):
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        y = tf.constant([1, 2, 4], dtype=tf.int32)
        z = not_equal(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [False, False, True]))

    def test_pow(self):
        x = tf.constant([1, 2, 3], dtype=tf.float32)
        y = tf.constant([2, 2, 2], dtype=tf.float32)
        z = pow(x, y)
        self.assertTrue(np.allclose(z.numpy(), np.power([1, 2, 3], [2, 2, 2])))

    def test_real(self):
        x = tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)
        y = real(x)
        self.assertTrue(np.array_equal(y.numpy(), [1, 3]))

    def test_reciprocal(self):
        x = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
        y = reciprocal(x)
        self.assertTrue(np.allclose(y.numpy(), 1 / np.array([1.0, 2.0, 4.0])))

    def test_reduce_prod(self):
        x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        y = reduce_prod(x)
        self.assertTrue(np.allclose(y.numpy(), np.prod([1, 2, 3, 4])))

    def test_reduce_std(self):
        x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        y = reduce_std(x)
        self.assertTrue(np.allclose(y.numpy(), np.std([1, 2, 3, 4])))

    def test_reduce_sum(self):
        x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        y = reduce_sum(x)
        self.assertTrue(np.allclose(y.numpy(), np.sum([1, 2, 3, 4])))

    def test_reduce_variance(self):
        x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        y = reduce_variance(x)
        self.assertTrue(np.allclose(y.numpy(), np.var([1, 2, 3, 4])))

    def test_round(self):
        x = tf.constant([0.5, 1.5, 2.5, 3.5], dtype=tf.float32)
        y = round(x)
        self.assertTrue(np.array_equal(y.numpy(), np.round([0.5, 1.5, 2.5, 3.5])))

    def test_rsqrt(self):
        x = tf.constant([1.0, 4.0, 9.0], dtype=tf.float32)
        y = rsqrt(x)
        self.assertTrue(np.allclose(y.numpy(), 1 / np.sqrt([1.0, 4.0, 9.0])))

    def test_segment_max(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
        y = segment_max(x, segment_ids)
        self.assertTrue(np.array_equal(y.numpy(), [[4, 5, 6], [7, 8, 9]]))

    def test_segment_mean(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
        y = segment_mean(x, segment_ids)
        self.assertTrue(np.allclose(y.numpy(), [[2.5, 3.5, 4.5], [7, 8, 9]]))

    def test_segment_min(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
        y = segment_min(x, segment_ids)
        self.assertTrue(np.array_equal(y.numpy(), [[1, 2, 3], [7, 8, 9]]))

    def test_segment_prod(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
        y = segment_prod(x, segment_ids)
        self.assertTrue(np.array_equal(y.numpy(), [[4, 10, 18], [7, 8, 9]]))

    def test_segment_sum(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
        y = segment_sum(x, segment_ids)
        self.assertTrue(np.array_equal(y.numpy(), [[5, 7, 9], [7, 8, 9]]))

    def test_sigmoid(self):
        x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        y = sigmoid(x)
        self.assertTrue(np.allclose(y.numpy(), 1 / (1 + np.exp(-np.array([0.0, 1.0, 2.0])))))

    def test_sign(self):
        x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
        y = sign(x)
        self.assertTrue(np.array_equal(y.numpy(), [-1.0, 0.0, 1.0]))

    def test_sin(self):
        x = tf.constant([0.0, np.pi / 2, np.pi], dtype=tf.float32)
        y = sin(x)
        self.assertTrue(np.allclose(y.numpy(), np.sin([0.0, np.pi / 2, np.pi])))

    def test_sinh(self):
        x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        y = sinh(x)
        self.assertTrue(np.allclose(y.numpy(), np.sinh([0.0, 1.0, 2.0])))

    def test_softplus(self):
        x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        y = softplus(x)
        self.assertTrue(np.allclose(y.numpy(), np.log1p(np.exp([0.0, 1.0, 2.0]))))

    def test_square(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = square(x)
        self.assertTrue(np.array_equal(y.numpy(), [1.0, 4.0, 9.0]))

    def test_squared_difference(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
        z = squared_difference(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [0.0, 0.0, 1.0]))

    def test_subtract(self):
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
        z = subtract(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [0.0, 0.0, -1.0]))

    def test_tan(self):
        x = tf.constant([0.0, np.pi / 4, np.pi / 2], dtype=tf.float32)
        y = tan(x)
        self.assertTrue(np.allclose(y.numpy(), np.tan([0.0, np.pi / 4, np.pi / 2])))

    def test_tanh(self):
        x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        y = tanh(x)
        self.assertTrue(np.allclose(y.numpy(), np.tanh([0.0, 1.0, 2.0])))

    def test_any(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = any(x)
        self.assertTrue(y.numpy())

    def test_all(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = all(x)
        self.assertFalse(y.numpy())

    def test_logical_and(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = tf.constant([True, True, False], dtype=tf.bool)
        z = logical_and(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [True, False, False]))

    def test_logical_or(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = tf.constant([True, True, False], dtype=tf.bool)
        z = logical_or(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [True, True, True]))

    def test_logical_not(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = logical_not(x)
        self.assertTrue(np.array_equal(y.numpy(), [False, True, False]))

    def test_logical_xor(self):
        x = tf.constant([True, False, True], dtype=tf.bool)
        y = tf.constant([True, True, False], dtype=tf.bool)
        z = logical_xor(x, y)
        self.assertTrue(np.array_equal(z.numpy(), [False, True, True]))

    def test_argsort(self):
        x = tf.constant([3, 1, 2], dtype=tf.int32)
        y = argsort(x)
        self.assertTrue(np.array_equal(y.numpy(), [1, 2, 0]))

    def test_bmm(self):
        x = tf.constant([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
        y = tf.constant([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
        res = bmm(x, y)
        expected = [[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]]
        self.assertTrue(np.array_equal(res.numpy(), expected))

    def test_where(self):
        condition = tf.constant([True, False, True, False])
        x = tf.constant([1, 2, 3, 4])
        y = tf.constant([5, 6, 7, 8])
        res = where(condition, x, y)
        self.assertTrue(np.array_equal(res.numpy(), [1, 6, 3, 8]))

    def test_ones_like(self):
        x = tf.constant([0.9, 0.1, 3.2, 1.2])
        res = ones_like(x, dtype=tf.int32)
        self.assertTrue(np.array_equal(res.numpy(), [1, 1, 1, 1]))

    def test_zeros_like(self):
        x = tf.constant([0.9, 0.1, 3.2, 1.2])
        res = zeros_like(x, dtype=tf.int32)
        self.assertTrue(np.array_equal(res.numpy(), [0, 0, 0, 0]))

    def test_squeeze(self):
        x = tf.ones(shape=[1, 2, 3])
        res = squeeze(x, axis=0)
        self.assertEqual(res.shape, [2, 3])

    def test_unsorted_segment_sum(self):
        x = tf.constant([1, 2, 3])
        res = unsorted_segment_sum(x, [0, 0, 1], num_segments=2)
        self.assertTrue(np.array_equal(res.numpy(), [3, 3]))

    def test_unsorted_segment_mean(self):
        x = tf.constant([1.0, 2.0, 3.0])
        res = unsorted_segment_mean(x, [0, 0, 1], num_segments=2)
        self.assertTrue(np.array_equal(res.numpy(), [1.5, 3.0]))

    def test_unsorted_segment_min(self):
        x = tf.constant([1.0, 2.0, 3.0])
        res = unsorted_segment_min(x, [0, 0, 1], num_segments=2)
        self.assertTrue(np.array_equal(res.numpy(), [1.0, 3.0]))

    def test_unsorted_segment_max(self):
        x = tf.constant([1.0, 2.0, 3.0])
        res = unsorted_segment_max(x, [0, 0, 1], num_segments=2)
        self.assertTrue(np.array_equal(res.numpy(), [2.0, 3.0]))

    def test_set_seed(self):
        set_seed(42)
        self.assertEqual(tf.random.uniform([1]).numpy(), tf.random.uniform([1]).numpy())

    def test_is_tensor(self):
        x = tf.constant([1, 2, 3])
        self.assertTrue(is_tensor(x))
        self.assertFalse(is_tensor([1, 2, 3]))

    def test_tensor_scatter_nd_update(self):
        tensor = tf.ones(shape=(5, 3))
        indices = [[0], [4], [2]]
        updates = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_tensor = tensor_scatter_nd_update(tensor, indices, updates)
        expected = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0], [4.0, 5.0, 6.0]]
        self.assertTrue(np.array_equal(new_tensor.numpy(), expected))

    def test_diag(self):
        tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_tensor = diag(tensor)
        self.assertTrue(np.array_equal(new_tensor.numpy(), [1, 5, 9]))

    def test_mask_select(self):
        tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mask = tf.constant([True, False, True])
        new_tensor = mask_select(tensor, mask)
        expected = [[1, 2, 3], [7, 8, 9]]
        self.assertTrue(np.array_equal(new_tensor.numpy(), expected))

    def test_eye(self):
        res = eye(2)
        expected = [[1, 0], [0, 1]]
        self.assertTrue(np.array_equal(res.numpy(), expected))

    def test_einsum(self):
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        res = einsum("i,j->ij", x, y)
        expected = [[4, 5, 6], [8, 10, 12], [12, 15, 18]]
        self.assertTrue(np.array_equal(res.numpy(), expected))

    def test_set_device(self):
        set_device("CPU")
        device = get_device()
        self.assertTrue("CPU" in device[0].device_type)

    # def test_scatter_update(self):
    #     x = tf.ones((5,))
    #     indices = tf.constant([0, 4, 2])
    #     updates = tf.constant([1.0, 4.0, 7.0])
    #     res = scatter_update(x, indices, updates)
    #     self.assertTrue(np.array_equal(res.numpy(), [1.0, 1.0, 7.0, 1.0, 4.0]))

    # def test_get_device(self):
    #     device = get_device()
    #     self.assertTrue("CPU" in device[0].device_type or "GPU" in device[0].device_type)

    # def test_to_device(self):
    #     x = tf.ones((5,))
    #     x = to_device(x, device="CPU", id=0)
    #     self.assertTrue(is_tensor(x))

    # def test_roll(self):
    #     x = tf.ones((5, 6))
    #     res = roll(x, shifts=2)
    #     self.assertTrue(np.array_equal(res.numpy(), np.roll(np.ones((5, 6)), 2)))

    def test_logsoftmax(self):
        x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        res = logsoftmax(x, dim=1)
        expected = tf.nn.log_softmax(x, axis=1)
        self.assertTrue(np.allclose(res.numpy(), expected.numpy()))

    def test_topk(self):
        x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        values, indices = topk(x, 2)
        self.assertTrue(np.array_equal(values.numpy(), [[3.0, 2.0], [3.0, 2.0]]))
        self.assertTrue(np.array_equal(indices.numpy(), [[2, 1], [2, 1]]))

    def test_numel(self):
        x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        self.assertEqual(numel(x).numpy(), 6)


if __name__ == "__main__":
    unittest.main()
