import jax
import jax.numpy as jnp
import pytest

from jaxragged import MaskedArray, ragged


@pytest.fixture
def simple_ma():
    return MaskedArray(
        data=jnp.array([4.0, 5.0, 0.0, 0.0]),
        mask=jnp.array([True, True, False, False]),
    )


@pytest.fixture
def negative_ma():
    return MaskedArray(
        data=jnp.array([-3.0, -1.0, 0.0, 0.0]),
        mask=jnp.array([True, True, False, False]),
    )


@pytest.fixture
def batch_ma():
    return MaskedArray.from_ragged([[1, 2, 3], [4, 5]])


class TestReductions:
    def test_sum(self, simple_ma):
        assert ragged(jnp.sum)(simple_ma) == 9.0

    def test_max(self, simple_ma):
        assert ragged(jnp.max)(simple_ma) == 5.0

    def test_min(self, simple_ma):
        assert ragged(jnp.min)(simple_ma) == 4.0

    def test_mean(self, simple_ma):
        assert ragged(jnp.mean)(simple_ma) == 4.5

    def test_std(self):
        ma = MaskedArray(
            data=jnp.array([2.0, 4.0, 0.0, 0.0]),
            mask=jnp.array([True, True, False, False]),
        )
        result = ragged(jnp.std)(ma)
        expected = jnp.std(jnp.array([2.0, 4.0]))
        assert jnp.allclose(result, expected)

    def test_max_with_negatives(self, negative_ma):
        assert ragged(jnp.max)(negative_ma) == -1.0

    def test_min_with_negatives(self, negative_ma):
        assert ragged(jnp.min)(negative_ma) == -3.0

    def test_mean_with_negatives(self, negative_ma):
        assert ragged(jnp.mean)(negative_ma) == -2.0


class TestCustomFunctions:
    def test_sum_times_two_plus_one(self):
        def f(x):
            return jnp.sum(x * 2) + 1

        ma = MaskedArray(
            data=jnp.array([1.0, 2.0, 3.0, 0.0, 0.0]),
            mask=jnp.array([True, True, True, False, False]),
        )
        assert ragged(f)(ma) == 13.0


class TestBatched:
    def test_auto_vmap_mean(self, batch_ma):
        result = ragged(jnp.mean)(batch_ma)
        assert jnp.allclose(result, jnp.array([2.0, 4.5]))

    def test_auto_vmap_sum(self, batch_ma):
        result = ragged(jnp.sum)(batch_ma)
        assert jnp.allclose(result, jnp.array([6.0, 9.0]))

    def test_auto_vmap_max(self, batch_ma):
        result = ragged(jnp.max)(batch_ma)
        assert jnp.allclose(result, jnp.array([3.0, 5.0]))

    def test_auto_vmap_min(self, batch_ma):
        result = ragged(jnp.min)(batch_ma)
        assert jnp.allclose(result, jnp.array([1.0, 4.0]))

    def test_auto_vmap_varied_lengths(self):
        ma = MaskedArray.from_ragged([[1], [2, 3, 4, 5], [10, 20]])
        result = ragged(jnp.mean)(ma)
        assert jnp.allclose(result, jnp.array([1.0, 3.5, 15.0]))

    def test_explicit_vmap_still_works(self, batch_ma):
        result = jax.vmap(ragged(jnp.mean))(batch_ma)
        assert jnp.allclose(result, jnp.array([2.0, 4.5]))


class TestJit:
    def test_jit(self, simple_ma):
        result = jax.jit(ragged(jnp.mean))(simple_ma)
        assert result == 4.5

    def test_jit_batched(self, batch_ma):
        result = jax.jit(ragged(jnp.mean))(batch_ma)
        assert jnp.allclose(result, jnp.array([2.0, 4.5]))


class TestFromRagged:
    def test_creates_correct_shape(self):
        ma = MaskedArray.from_ragged([[1, 2, 3], [4, 5]])
        assert ma.data.shape == (2, 3)
        assert ma.mask.shape == (2, 3)

    def test_creates_correct_mask(self):
        ma = MaskedArray.from_ragged([[1, 2, 3], [4, 5]])
        expected_mask = jnp.array([[True, True, True], [True, True, False]])
        assert jnp.array_equal(ma.mask, expected_mask)

    def test_creates_correct_data(self):
        ma = MaskedArray.from_ragged([[1, 2, 3], [4, 5]])
        expected_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        assert jnp.array_equal(ma.data, expected_data)
