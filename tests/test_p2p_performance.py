import numpy as np
import pytest


def test_p2p_model_counts_pair_loop_and_caches_source_analysis(tmp_path, monkeypatch):
    from parkipy.ewald import _perf

    source = tmp_path / '_pk_kernels' / '_p2p_workunits.py'
    source.parent.mkdir()
    source.write_text('''\ndef helper(a):\n    return a * a + pk.erf(a)\ndef p2p_stokes_sl_gm1d_fp64():\n    outside = 1.0 * 2.0\n    for s in range(5):\n        x = source[2 * s] - target[0]\n        x += helper(x)\n''')
    monkeypatch.setattr(_perf, '__file__', str(tmp_path / '_perf.py'))
    _perf._p2p_operation_counts.cache_clear()
    try:
        counts = dict(_perf._p2p_operation_counts('stokes_sl'))
        assert counts == {'fadd': 3, 'fmul': 1, 'ferf': 1}
        expected = 2 * 3 * (4 * np.pi / 3) * 5
        assert _perf.PerfModel._count_flop_p2p(None, 'stokes_sl', 2, 3, None) == pytest.approx(expected)
        source.write_text('invalid source')
        assert dict(_perf._p2p_operation_counts('stokes_sl')) == counts
        assert _perf._p2p_operation_counts.cache_info().hits >= 2
        _perf._p2p_operation_counts.cache_clear()
        with pytest.raises(SyntaxError):
            _perf._p2p_operation_counts('stokes_sl')
    finally:
        _perf._p2p_operation_counts.cache_clear()
