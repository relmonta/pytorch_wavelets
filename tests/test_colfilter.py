import numpy as np
from pytorch_wavelets.dtcwt.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter

import pytest
import datasets
from pytorch_wavelets.dtcwt.lowlevel import colfilter, prep_filt
import torch
import py3nvml

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixture(scope="module") ensures this function is computed once and shared among
# all test functions in the same file.
# It is not recommended to name fixtures "setup" in pytest as it is
# generally used unittest-style


@pytest.fixture(scope="module")
def barbara_data():
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara / barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extrarow = bshape[:]
    bshape_extrarow[1] += 1
    barbara_t = torch.unsqueeze(torch.tensor(barbara), dim=0).to(dev)
    ch = barbara_t.shape[1]

    def ref_colfilter(x, h): return np.stack(
        [np_colfilter(s, h) for s in x], axis=0)

    return {
        "barbara": barbara,
        "barbara_t": barbara_t,
        "bshape": bshape,
        "bshape_extrarow": bshape_extrarow,
        "ref_colfilter": ref_colfilter,
        "ch": ch,
    }


def test_barbara_loaded(barbara_data):
    barbara = barbara_data["barbara"]
    barbara_t = barbara_data["barbara_t"]

    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    bshape = barbara_data["bshape"]
    h = [-1, 2, -1]
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape


def test_even_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    bshape_extrarow = barbara_data["bshape_extrarow"]
    h = [-1, -1]
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_qshift(barbara_data):
    h = qshift('qshift_a')[0]
    y_op = colfilter(barbara_data["barbara_t"], prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == barbara_data["bshape_extrarow"]


def test_biort(barbara_data):
    h = biort('antonini')[0]
    y_op = colfilter(barbara_data["barbara_t"], prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == barbara_data["bshape"]


def test_even_size_batch(barbara_data):
    barbara = barbara_data["barbara"]
    bshape_extrarow = barbara_data["bshape_extrarow"]
    zero_t = torch.zeros((1, *barbara.shape), dtype=torch.float32).to(dev)
    y = colfilter(zero_t, prep_filt([-1, 1], 1).to(dev))
    assert list(y.shape)[1:] == bshape_extrarow
    assert not np.any(y.cpu().numpy()[:] != 0.0)


def test_equal_small_in(barbara_data):
    barbara = barbara_data["barbara"]
    ref_colfilter = barbara_data["ref_colfilter"]
    h = qshift('qshift_b')[0]
    im = barbara[:, 0:4, 0:4]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort1(barbara_data):
    barbara = barbara_data["barbara"]
    ref_colfilter = barbara_data["ref_colfilter"]
    h = biort('near_sym_b')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_data["barbara_t"], prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort2(barbara_data):
    barbara = barbara_data["barbara"]
    ref_colfilter = barbara_data["ref_colfilter"]
    h = biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift1(barbara_data):
    barbara = barbara_data["barbara"]
    ref_colfilter = barbara_data["ref_colfilter"]
    h = qshift('qshift_c')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_data["barbara_t"], prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift2(barbara_data):
    barbara = barbara_data["barbara"]
    ref_colfilter = barbara_data["ref_colfilter"]
    h = qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip(reason="gradient check is unstable or optional")
def test_gradients(barbara_data):
    barbara = barbara_data["barbara"]
    h = biort('near_sym_b')[0]
    im_t = torch.unsqueeze(torch.tensor(
        barbara, dtype=torch.float32, requires_grad=True), dim=0)
    y_t = colfilter(im_t, prep_filt(h, 1))
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))
