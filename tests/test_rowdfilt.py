import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt
from pytorch_wavelets.dtcwt.lowlevel import rowdfilt, prep_filt
import torch
import py3nvml
import pytest
from pytest import raises
import datasets

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def barbara_data():
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[2] //= 2
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0).to(dev)
    # Some useful functions
    def ref_rowdfilt(x, ha, hb): return np.stack(
        [np_coldfilt(s.T, ha, hb).T for s in x], axis=0)

    return {
        "barbara": barbara,
        "barbara_t": barbara_t,
        "bshape": bshape,
        "bshape_half": bshape_half,
        "ref_rowdfilt": ref_rowdfilt
    }


def test_barbara_loaded(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    barbara = barbara_data["barbara"]
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


@pytest.mark.skip
def test_odd_filter(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-1, 2, -1), 1).to(dev)
        hb = prep_filt((-1, 2, 1), 1).to(dev)
        rowdfilt(barbara_t, ha, hb)


@pytest.mark.skip
def test_different_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-0.5, -1, 2, 0.5), 1).to(dev)
        hb = prep_filt((-1, 2, 1), 1).to(dev)
        rowdfilt(barbara_t, ha, hb)


def test_bad_input_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    with raises(ValueError):
        ha = prep_filt((-1, 1), 1).to(dev)
        hb = prep_filt((1, -1), 1).to(dev)
        rowdfilt(barbara_t[:, :, :, :511], ha, hb)


def test_good_input_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    barbara = barbara_data["barbara"]
    ha = prep_filt((-1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    rowdfilt(barbara_t[:, :, :511, :], ha, hb)


def test_good_input_size_non_orthogonal(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    ha = prep_filt((1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    rowdfilt(barbara_t[:, :, :511, :], ha, hb)


def test_output_size(barbara_data):
    barbara_t = barbara_data["barbara_t"]
    bshape_half = barbara_data["bshape_half"]
    ha = prep_filt((-1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    y_op = rowdfilt(barbara_t, ha, hb)
    assert list(y_op.shape[1:]) == bshape_half


@pytest.mark.parametrize('hp', [False, True])
def test_equal_small_in(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_rowdfilt = barbara_data["ref_rowdfilt"]
    ha, hb = qshift('qshift_b')[4:6] if hp else qshift('qshift_b')[0:2]
    im = barbara[:, 0:4, 0:4]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift1(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_rowdfilt = barbara_data["ref_rowdfilt"]
    barbara_t = barbara_data["barbara_t"]
    ha, hb = qshift('qshift_a')[4:6] if hp else qshift('qshift_a')[0:2]
    ref = ref_rowdfilt(barbara, ha, hb)
    y = rowdfilt(barbara_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift2(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_rowdfilt = barbara_data["ref_rowdfilt"]
    ha, hb = qshift('qshift_b')[4:6] if hp else qshift('qshift_b')[0:2]
    im = barbara[:, :502, :508]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift3(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_rowdfilt = barbara_data["ref_rowdfilt"]
    ha, hb = qshift('qshift_c')[4:6] if hp else qshift('qshift_c')[0:2]
    im = barbara[:, :502, :508]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift4(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ref_rowdfilt = barbara_data["ref_rowdfilt"]
    ha, hb = qshift('qshift_d')[4:6] if hp else qshift('qshift_d')[0:2]
    im = barbara[:, :502, :508]
    im_t = torch.unsqueeze(torch.tensor(
        im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip
def test_gradients(barbara_data, hp):
    barbara = barbara_data["barbara"]
    ha, hb = qshift('qshift_b')[4:6] if hp else qshift('qshift_b')[0:2]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0).to(dev)
    y_t = rowdfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                   highpass=hp)
    np.random.randn(*tuple(y_t.shape)).astype('float32')
