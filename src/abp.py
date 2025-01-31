# -*- coding: utf-8 -*-
"""
biosppy.signals.abp
-------------------

This module provides methods to process Arterial Blood Pressure (ABP) signals
and extract relevant features such as systolic pressure, diastolic pressure,
mean arterial pressure (MAP), and heart rate variability (HRV) metrics.
"""

# Imports
from __future__ import absolute_import, division, print_function
from six.moves import range

import numpy as np

from . import tools as st
from .. import plotting, utils


def abp(signal=None, sampling_rate=1000.0, show=True):
    """Process a raw ABP signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ABP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ABP signal.
    onsets : array
        Indices of ABP pulse onsets.
    systolic_peaks : array
        Indices of systolic peaks.
    diastolic_troughs : array
        Indices of diastolic troughs.
    systolic_pressure : array
        Systolic pressure values.
    diastolic_pressure : array
        Diastolic pressure values.
    mean_arterial_pressure : array
        Mean arterial pressure (MAP) values.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy array
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="bandpass",
        order=4,
        frequency=[0.5, 15],
        sampling_rate=sampling_rate,
    )

    # find onsets
    (onsets,) = find_onsets_zong2003(signal=filtered, sampling_rate=sampling_rate)

    # find systolic and diastolic points
    systolic_peaks, diastolic_troughs = find_systolic_diastolic(
        signal=filtered, onsets=onsets
    )

    # extract pressure values
    systolic_pressure = signal[systolic_peaks]
    diastolic_pressure = signal[diastolic_troughs]

    # compute mean arterial pressure (MAP)
    mean_arterial_pressure = compute_map(systolic_pressure, diastolic_pressure)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(
        beats=onsets, sampling_rate=sampling_rate, smooth=True, size=3
    )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    ts_hr = ts[hr_idx]

    # plot
    if show:
        plotting.plot_abp(
            ts=ts,
            raw=signal,
            filtered=filtered,
            onsets=onsets,
            systolic_peaks=systolic_peaks,
            diastolic_troughs=diastolic_troughs,
            heart_rate_ts=ts_hr,
            heart_rate=hr,
            path=None,
            show=True,
        )

    # output
    args = (
        ts,
        filtered,
        onsets,
        systolic_peaks,
        diastolic_troughs,
        systolic_pressure,
        diastolic_pressure,
        mean_arterial_pressure,
        ts_hr,
        hr,
    )
    names = (
        "ts",
        "filtered",
        "onsets",
        "systolic_peaks",
        "diastolic_troughs",
        "systolic_pressure",
        "diastolic_pressure",
        "mean_arterial_pressure",
        "heart_rate_ts",
        "heart_rate",
    )

    return utils.ReturnTuple(args, names)

def find_onsets_zong2003(
    signal=None,
    sampling_rate=1000.0,
    sm_size=None,
    size=None,
    alpha=2.0,
    wrange=None,
    d1_th=0,
    d2_th=None,
):
    """Determine onsets of ABP pulses.

    Skips corrupted signal parts.
    Based on the approach by Zong *et al.* [Zong03]_.

    Parameters
    ----------
    signal : array
        Input filtered ABP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    sm_size : int, optional
        Size of smoother kernel (seconds).
        Defaults to 0.25
    size : int, optional
        Window to search for maxima (seconds).
        Defaults to 5
    alpha : float, optional
        Normalization parameter.
        Defaults to 2.0
    wrange : int, optional
        The window in which to search for a peak (seconds).
        Defaults to 0.1
    d1_th : int, optional
        Smallest allowed difference between maxima and minima.
        Defaults to 0
    d2_th : int, optional
        Smallest allowed time between maxima and minima (seconds),
        Defaults to 0.15

    Returns
    -------
    onsets : array
        Indices of ABP pulse onsets.

    References
    ----------
    .. [Zong03] W Zong, T Heldt, GB Moody and RG Mark, "An Open-source
       Algorithm to Detect Onset of Arterial Blood Pressure Pulses",
       IEEE Comp. in Cardiology, vol. 30, pp. 259-262, 2003
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # parameters
    sm_size = 0.25 if not sm_size else sm_size
    sm_size = int(sm_size * sampling_rate)
    size = 5 if not size else size
    size = int(size * sampling_rate)
    wrange = 0.1 if not wrange else wrange
    wrange = int(wrange * sampling_rate)
    d2_th = 0.15 if not d2_th else d2_th
    d2_th = int(d2_th * sampling_rate)

    length = len(signal)

    # slope sum function
    dy = np.diff(signal)
    dy[dy < 0] = 0

    ssf, _ = st.smoother(signal=dy, kernel="boxcar", size=sm_size, mirror=True)

    # main loop
    start = 0
    stop = size
    if stop > length:
        stop = length

    idx = []

    while True:
        sq = np.copy(signal[start:stop])
        sq -= sq.mean()
        ss = 25 * ssf[start:stop]
        sss = 100 * np.diff(ss)
        sss[sss < 0] = 0
        sss = sss - alpha * np.mean(sss)

        # find maxima
        pk, pv = st.find_extrema(signal=sss, mode="max")
        pk = pk[np.nonzero(pv > 0)]
        pk += wrange
        dpidx = pk

        # analyze between maxima of 2nd derivative of ss
        detected = False
        for i in range(1, len(dpidx) + 1):
            try:
                v, u = dpidx[i - 1], dpidx[i]
            except IndexError:
                v, u = dpidx[-1], -1

            s = sq[v:u]
            Mk, Mv = st.find_extrema(signal=s, mode="max")
            mk, mv = st.find_extrema(signal=s, mode="min")

            try:
                M = Mk[np.argmax(Mv)]
                m = mk[np.argmin(mv)]
            except ValueError:
                continue

            if (s[M] - s[m] > d1_th) and (m - M > d2_th):
                idx += [v + start]
                detected = True

        # next round continues from previous detected beat
        if detected:
            start = idx[-1] + wrange
        else:
            start += size

        # stop condition
        if start > length:
            break

        # update stop
        stop += size
        if stop > length:
            stop = length

    idx = np.array(idx, dtype="int")

    return utils.ReturnTuple((idx,), ("onsets",))


def find_systolic_diastolic(signal, onsets):
    """Find systolic peaks and diastolic troughs in the ABP signal.

    Parameters
    ----------
    signal : array
        Filtered ABP signal.
    onsets : array
        Indices of ABP pulse onsets.

    Returns
    -------
    systolic_peaks : array
        Indices of systolic peaks.
    diastolic_troughs : array
        Indices of diastolic troughs.
    """
    systolic_peaks = []
    diastolic_troughs = []

    num_beats = len(onsets)
    for i in range(num_beats - 1):
        start = onsets[i]
        end = onsets[i + 1]

        # Extract one pulse
        pulse = signal[start:end]

        # Find systolic peak within the pulse
        systolic_index = np.argmax(pulse) + start
        systolic_peaks.append(systolic_index)

        # Find diastolic trough within the pulse
        diastolic_index = np.argmin(pulse) + start
        diastolic_troughs.append(diastolic_index)

    # Handle last beat
    if num_beats >= 2:
        start = onsets[-1]
        end = len(signal)

        pulse = signal[start:end]
        if len(pulse) > 0:
            systolic_index = np.argmax(pulse) + start
            systolic_peaks.append(systolic_index)

            diastolic_index = np.argmin(pulse) + start
            diastolic_troughs.append(diastolic_index)

    return np.array(systolic_peaks), np.array(diastolic_troughs)


def compute_map(systolic_pressure, diastolic_pressure):
    """Compute Mean Arterial Pressure (MAP).

    Parameters
    ----------
    systolic_pressure : array
        Systolic pressure values.
    diastolic_pressure : array
        Diastolic pressure values.

    Returns
    -------
    mean_arterial_pressure : array
        Mean arterial pressure values.
    """
    # Using the formula:
    # MAP = Diastolic + 1/3 * (Systolic - Diastolic)
    mean_arterial_pressure = diastolic_pressure + (systolic_pressure - diastolic_pressure) / 3.0
    return mean_arterial_pressure