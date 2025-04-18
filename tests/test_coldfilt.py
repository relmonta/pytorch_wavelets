from pytest import raises
import pytest

import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt
import datasets
from pytorch_wavelets.dtcwt.lowlevel import coldfilt, prep_filt
import torch
import py3nvml

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def barbara_data():
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara / barbara.max()).astype('float32').transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(torch.tensor(
        barbara, dtype=torch.float32, device=dev), dim=0)
    ch = barbara_t.shape[1]

    def ref_coldfilt(x, ha, hb):
        return np.stack([np_coldfilt(s, ha, hb) for s in x], axis=0)

    return {
        "barbara": barbara,
        "barbara_t": barbara_t,
        "bshape": bshape,
        "bshape_half": bshape_half,
        "ref_coldfilt": ref_coldfilt,
        "ch": ch
    }


def test_barbara_loaded(barbara_data):
    barbara = barbara_data["barbara"]
    barbara_t = barbara_data["barbara_t"]

    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_output_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    bshape_half = barbara_data["bshape_half"]
    ha = prep_filt([-1, 1], 1).to(dev)
    hb = prep_filt([1, -1], 1).to(dev)
    y_op = coldfilt(barbara_t, ha, hb)
    assert list(y_op.shape)[1:] == bshape_half


@pytest.mark.skip("Don't currently check for this in lowlevel code for speed")
def test_odd_filter(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-1, 2, -1), 1).to(dev)
        hb = prep_filt((-1, 2, 1), 1).to(dev)
        coldfilt(barbara_t, ha, hb)


@pytest.mark.skip("Don't currently check for this in lowlevel code for speed")
def test_different_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-0.5, -1, 2, 0.5), 1).to(dev)
        hb = prep_filt((-1, 2, 1), 1).to(dev)
        coldfilt(barbara_t, ha, hb)


def test_bad_input_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-1, 1), 1).to(dev)
        hb = prep_filt((1, -1), 1).to(dev)
        coldfilt(barbara_t[:, :, :511, :], ha, hb)


def test_good_input_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    ha = prep_filt((-1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    coldfilt(barbara_t[:, :, :, :511], ha, hb)


def test_good_input_size_non_orthogonal(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    ha = prep_filt((1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    coldfilt(barbara_t[:, :, :, :511], ha, hb)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_small_in(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_coldfilt = barbara_data["ref_coldfilt"]

    ha, hb = qshift('qshift_a')[4:6] if hp else qshift('qshift_a')[0:2]
    im = barbara[:, 0:4, 0:4]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev),
                 prep_filt(hb, 1).to(dev), highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift1(barbara_data, hp):
    barbara = barbara_data["barbara"]
    barbara_t = barbara_data["barbara_t"]
    ref_coldfilt = barbara_data["ref_coldfilt"]

    ha, hb = qshift('qshift_a')[4:6] if hp else qshift('qshift_a')[0:2]
    ref = ref_coldfilt(barbara, ha, hb)
    y = coldfilt(barbara_t, prep_filt(ha, 1).to(dev),
                 prep_filt(hb, 1).to(dev), highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift2(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_coldfilt = barbara_data["ref_coldfilt"]

    ha, hb = qshift('qshift_a')[4:6] if hp else qshift('qshift_a')[0:2]
    im = barbara[:, :508, :502]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev),
                 prep_filt(hb, 1).to(dev), highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift3(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_coldfilt = barbara_data["ref_coldfilt"]

    ha, hb = qshift('qshift_a')[4:6] if hp else qshift('qshift_a')[0:2]
    im = barbara[:, :508, :502]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev),
                 prep_filt(hb, 1).to(dev), highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip
def test_gradients(barbara_data):
    barbara = barbara_data["barbara"]
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = coldfilt(im_t, prep_filt(ha, 1), prep_filt(hb, 1), np.sum(ha*hb) > 0)
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))
