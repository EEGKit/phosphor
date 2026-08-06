"""SweepBuffer's CPU-side reduction, including the envelope input mode.

``SweepBuffer`` is pure numpy and threading -- no canvas, no GPU -- so it can be
exercised headlessly, which is where the reduction logic actually lives.

The property that matters for envelope mode is *equivalence*: pushing a
pre-reduced (min, max) stream must draw what the raw signal would have drawn.
If that does not hold, moving decimation upstream changes what the user sees,
and the whole point was that it should not.
"""

import numpy as np
import pytest

from phosphor.sweep_buffer import SweepBuffer


def make_buffer(**kwargs) -> SweepBuffer:
    defaults = dict(n_channels=4, srate=1000.0, display_dur=1.0, n_columns=10, n_visible=4)
    defaults.update(kwargs)
    return SweepBuffer(**defaults)


def minmax_decimate(raw: np.ndarray, factor: int) -> np.ndarray:
    """(n, ch) -> (n // factor, ch, 2), the shape an upstream decimator emits."""
    n = raw.shape[0] // factor * factor
    buckets = raw[:n].reshape(-1, factor, raw.shape[1])
    return np.stack([buckets.min(axis=1), buckets.max(axis=1)], axis=-1)


# ---- raw mode is unchanged --------------------------------------------------


def test_raw_push_reduces_to_columns():
    buf = make_buffer()
    raw = np.zeros((1000, 4), dtype=np.float32)
    raw[15, 0] = 5.0  # column 0 spans samples 0..99
    raw[150, 1] = -3.0  # column 1 spans 100..199
    buf.push_data(raw)

    assert buf.display_maxs[0, 0] == pytest.approx(5.0)
    assert buf.display_mins[1, 1] == pytest.approx(-3.0)


def test_raw_push_still_accepts_1d():
    buf = make_buffer(n_channels=1, n_visible=1)
    buf.push_data(np.ones(1000, dtype=np.float32))
    assert buf.display_maxs.max() == pytest.approx(1.0)


# ---- envelope mode ----------------------------------------------------------


def test_envelope_matches_raw_for_the_same_signal():
    """The equivalence that justifies decimating upstream at all.

    Note the envelope buffer is configured at the *envelope* rate, not the raw
    one: ``srate`` describes the stream being pushed. Both buffers then span the
    same wall-clock second over the same ten columns, so each column covers the
    same 100 raw samples, and the two reductions must agree exactly.
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((1000, 4)).astype(np.float32)
    factor = 10

    from_raw = make_buffer(srate=1000.0)
    from_raw.push_data(raw)

    from_env = make_buffer(srate=1000.0 / factor, envelope=True)
    from_env.push_data(minmax_decimate(raw, factor))

    np.testing.assert_allclose(from_env.display_mins, from_raw.display_mins)
    np.testing.assert_allclose(from_env.display_maxs, from_raw.display_maxs)


def test_envelope_srate_is_the_bucket_rate():
    """Sizing from the pre-decimation rate is the easy mistake: the ring would
    be `factor` times too long and the sweep would sit mostly empty."""
    buf = make_buffer(srate=100.0, display_dur=1.0, envelope=True)
    assert buf.total_raw_samples == 100
    assert buf.raw_buffer.shape == (100, 4, 2)


def test_envelope_preserves_a_spike_stride_decimation_would_lose():
    """The motivating case, end to end through the buffer."""
    raw = np.zeros((1000, 1), dtype=np.float32)
    raw[37, 0] = 100.0  # missed by raw[::10]

    buf = make_buffer(n_channels=1, n_visible=1, envelope=True)
    buf.push_data(minmax_decimate(raw, 10))

    assert buf.display_maxs.max() == pytest.approx(100.0)


def test_envelope_allocates_a_rank_3_raw_buffer():
    assert make_buffer(envelope=True).raw_buffer.shape == (1000, 4, 2)
    assert make_buffer().raw_buffer.shape == (1000, 4)


def test_envelope_rejects_raw_shaped_data():
    """A silent misread here would plot half the channels at wrong values."""
    buf = make_buffer(envelope=True)
    with pytest.raises(ValueError, match="expects .*n_channels, 2"):
        buf.push_data(np.zeros((100, 4), dtype=np.float32))


def test_raw_mode_rejects_envelope_shaped_data():
    buf = make_buffer()
    with pytest.raises(ValueError, match="Did you mean envelope=True"):
        buf.push_data(np.zeros((100, 4, 2), dtype=np.float32))


def test_envelope_channel_padding_keeps_the_pair_axis():
    """Fewer channels than configured pads the channel axis only."""
    buf = make_buffer(n_channels=4, n_visible=4, envelope=True)
    data = np.ones((100, 2, 2), dtype=np.float32)
    buf.push_data(data)
    # Channels 0-1 carry the pushed value; 2-3 were padded with zeros.
    assert buf.display_maxs[0, 0] == pytest.approx(1.0)
    assert buf.display_maxs[0, 3] == pytest.approx(0.0)


def test_envelope_channel_trimming():
    buf = make_buffer(n_channels=2, n_visible=2, envelope=True)
    buf.push_data(np.ones((100, 5, 2), dtype=np.float32))
    assert buf.display_maxs.shape[1] == 2


def test_envelope_wraps_the_ring_like_raw():
    """More samples than one sweep must wrap, not overflow."""
    buf = make_buffer(envelope=True)
    env = np.zeros((1500, 4, 2), dtype=np.float32)
    env[..., 1] = 2.0
    buf.push_data(env)  # 1.5x the 1000-sample ring
    assert buf.display_maxs.max() == pytest.approx(2.0)


def test_set_envelope_reallocates():
    buf = make_buffer()
    assert buf.raw_buffer.ndim == 2
    buf.set_envelope(True)
    assert buf.raw_buffer.ndim == 3
    assert buf.envelope
    # Idempotent: no reallocation, and the version does not churn.
    version = buf.version
    buf.set_envelope(True)
    assert buf.version == version


def test_envelope_scale_and_midpoint_use_both_bounds():
    """_compute_y_scale/_compute_ch_mid read display_mins/maxs, so an envelope
    must fill both -- a bug filling only one would autoscale to half range."""
    buf = make_buffer(n_channels=1, n_visible=1, envelope=True)
    env = np.zeros((100, 1, 2), dtype=np.float32)
    env[..., 0] = -4.0
    env[..., 1] = 4.0
    buf.push_data(env)

    assert buf.display_mins.min() == pytest.approx(-4.0)
    assert buf.display_maxs.max() == pytest.approx(4.0)
    # ±0.5 normalization over a ±4 range.
    assert buf._compute_y_scale() == pytest.approx(0.125)
