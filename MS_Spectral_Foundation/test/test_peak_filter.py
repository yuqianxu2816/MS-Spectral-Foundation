import numpy as np
import pytest



from MS_Spectral_Foundation import peak_filter as pf


def test_to_array_none_and_empty_and_bad_shape():
    assert pf._to_array(None).shape == (0, 2)
    assert pf._to_array([]).shape == (0, 2)

    with pytest.raises(ValueError):
        pf._to_array([1, 2, 3])  # testing whether is a 2D array

    with pytest.raises(ValueError):
        pf._to_array([[1, 2, 3]])  # is a two D array, but (N,3) is wrong, should be (N, 2), two columns for mz and intensity, not three


def test_set_mz_range_noop_and_swap_and_infer():
    peaks = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]])

    # mz_min and mz_max both None -> infer from data branch -> keep all 3 peaks
    fn = pf.set_mz_range(None, None)
    out = fn(peaks)
    assert out.shape[0] == 3

    # max < min -> swap
    fn2 = pf.set_mz_range(250.0, 150.0)
    out2 = fn2(peaks)
    # keep in [150,250] => only 200
    assert out2.shape[0] == 1
    assert out2[0, 0] == 200.0

    # inclusive range:
    fn3 = pf.set_mz_range(100.0, 200.0)
    out3 = fn3(peaks)
    assert out3.shape[0] == 2
    assert set(out3[:, 0].tolist()) == {100.0, 200.0}


def test_remove_precursor_peak_invalid_unit_raises():
    with pytest.raises(ValueError):
        pf.remove_precursor_peak(1.0, unit="meter")  # unit: not in {"Da","ppm"}


def test_remove_precursor_peak_branch_no_precursor_and_with_precursor_Da():
    peaks = np.array([[99.0, 1.0], [100.0, 2.0], [101.0, 3.0]])

    fn = pf.remove_precursor_peak(tol=0.5, unit="Da")

    # precursor_mz missing -> keep all 3 peaks (unable to recognize precursor peak)
    out_no = fn(peaks, precursor_mz=None)
    assert out_no.shape[0] == 3

    # with precursor: |mz-100|<=0.5 will be removed, which mz=100 is removed
    out_yes = fn(peaks, precursor_mz=100.0)
    assert set(out_yes[:, 0].tolist()) == {99.0, 101.0}

def test_remove_precursor_peak_ppm_unit():
    peaks = np.array([
        [999.98, 1.0],   # |999.98 - 1000| = 0.02 > 0.01, keep
        [999.995, 2.0],  # |999.995 - 1000| = 0.005 <= 0.01, remove
        [1000.0, 3.0],   # |1000 - 1000| = 0 <= 0.01, remove
        [1000.005, 4.0], # |1000.005 - 1000| = 0.005 <= 0.01, remove
        [1000.02, 5.0],  # |1000.02 - 1000| = 0.02 > 0.01, keep
    ])
    
    fn = pf.remove_precursor_peak(tol=10, unit="ppm")
    out = fn(peaks, precursor_mz=1000.0)
    
    # Should keep only peaks at 999.98 and 1000.02
    assert out.shape[0] == 2
    assert set(out[:, 0].tolist()) == {999.98, 1000.02}

def test_scale_intensity_none_root_log_rank_and_errors():
    peaks = np.array([[100.0, -1.0], [200.0, 4.0], [300.0, 9.0]])

    # no scaling: -1 -> 0 
    fn_none = pf.scale_intensity(None)
    out_none = fn_none(peaks)
    assert out_none[0, 1] == 0.0

    # root: sqrt(4)=2
    fn_root = pf.scale_intensity("root", degree=2)
    out_root = fn_root(peaks)
    assert abs(out_root[1, 1] - 2.0) < 1e-9
    assert abs(out_root[2, 1] - 3.0) < 1e-9

    # log: log2(inten+1)
    fn_log = pf.scale_intensity("log", base=2)
    out_log = fn_log(np.array([[100.0, 3.0]]))
    assert abs(out_log[0, 1] - 2.0) < 1e-9

    # rank: highest intensity has the highest rank
    fn_rank = pf.scale_intensity("rank")
    out_rank = fn_rank(np.array([[1.0, 10.0], [2.0, 1.0], [3.0, 5.0]]))
    # The peak with intensity=10 should have the highest rank
    max_rank_idx = np.argmax(out_rank[:, 1])
    assert out_rank[max_rank_idx, 0] == 1.0

    with pytest.raises(ValueError):
        pf.scale_intensity("rank", max_rank=2)(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))

    with pytest.raises(ValueError):
        pf.scale_intensity("weird")(peaks)


def test_filter_intensity_threshold_max_peaks_and_sorting():
    # Intensities with <=0 will be discarded first 
    peaks = np.array([
        [300.0, 10.0],
        [100.0, 0.0],
        [200.0, 1.0],
        [150.0, 5.0],
    ])

    # keep intensities > min_intensity * max_intensity,
    # 0.2*10=2.0, so only 10 and 5 are kept
    fn = pf.filter_intensity(min_intensity=0.2, max_peaks=None)
    out = fn(peaks)
    assert set(out[:, 0].tolist()) == {150.0, 300.0}

    # max_peaks=1: only keep the strongest peak of intensity = 10
    fn2 = pf.filter_intensity(min_intensity=0.0, max_peaks=1)
    out2 = fn2(peaks)
    assert out2.shape[0] == 1
    assert out2[0, 1] == 10.0 

    # sorted by mz
    fn3 = pf.filter_intensity(min_intensity=0.0, max_peaks=None)
    out3 = fn3(peaks)
    assert np.all(out3[:, 0] == np.sort(out3[:, 0]))


def test_discard_low_quality_branch():
    # At least 3 peaks are required, so 2 peaks should be discarded
    peaks = np.array([[100.0, 1.0], [200.0, 2.0]])
    fn = pf.discard_low_quality(min_peaks=3)
    assert fn(peaks) is None

    # At least 2 peaks are required, so 2 peaks should be kept
    fn2 = pf.discard_low_quality(min_peaks=2)
    out = fn2(peaks)
    assert out is not None
    assert out.shape[0] == 2



def test_scale_to_unit_norm_zero_norm_and_nonzero():
    # All zero intensities -> norm=0
    # L2 norm = srqt(0^2+0^2)=0 -> return original peaks without scaling
    peaks0 = np.array([[100.0, 0.0], [200.0, 0.0]])
    out0 = pf._scale_to_unit_norm(peaks0)
    assert np.all(out0[:, 1] == 0.0)

    # Non-zero: L2 norm should be 1
    # norm = sqrt(3^2+4^2)=5 -> scaled intensities should be 3/5=0.6 and 4/5=0.8
    # sqrt(0.6^2+0.8^2)=1 -> unit norm achieved
    peaks = np.array([[100.0, 3.0], [200.0, 4.0]])
    out = pf._scale_to_unit_norm(peaks)
    norm = np.sqrt(np.sum(out[:, 1] ** 2))
    assert abs(norm - 1.0) < 1e-9


def test_apply_preprocessing_pipeline_typeerror_fallback_branch():
    # pipeline contain both: functions require precursor_mz (2 args) and not require (1 arg)
    peaks = np.array([[100.0, 1.0], [101.0, 2.0]])

    fn_need_prec = pf.remove_precursor_peak(tol=0.1, unit="Da")  # (peaks, precursor_mz)
    fn_no_prec = pf.set_mz_range(100.0, 200.0)                   # (peaks)

    out = pf.apply_preprocessing_pipeline(peaks, [fn_need_prec, fn_no_prec], precursor_mz=None)
    # When precursor=None, remove_precursor_peak does nothing; set_mz_range keeps both peaks
    assert out is not None
    assert out.shape[0] == 2
