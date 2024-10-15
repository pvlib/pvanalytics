"""Quality control functions for energy data."""
import warnings


def cumulative_energy_simple_diff_check(energy_series,
                                        pct_increase_threshold=95,
                                        system_self_consumption=-0.5):
    """Check if an energy time series is cumulative via simple differencing.

    To determine if an energy data stream is cumulative, subsequent values in
    the series are differenced to determine if the data stream is consistently
    increasing. If the percentage of increasing values in the data set exceeds
    the pct_increase_threshold parameter, the energy series is determined as
    cumulative and a True boolean is returned. Otherwise, False is returned.

    Parameters
    ----------
    energy_series: Series
        Time series of energy data stream with datetime index.
    pct_increase_threshold: Int, default 95
        The percentage threshold to consider the energy series as cumulative.
    system_self_consumption: Float, default -0.5
        The difference threshold to account for the effect of nighttime system
        self-consumption on the energy data stream.

    Returns
    -------
    Boolean
        True if energy_series is cumulative, False otherwise.
    """
    # Simple diff function
    differenced_series = energy_series.diff().dropna()
    if len(differenced_series) == 0:
        warnings.warn(
            "The energy time series has a length of zero and cannot be run.")
        return False
    else:
        # If over X percent of the data is increasing (set via the
        # pct_increase_threshold), then assume that the column is cumulative
        differenced_series_positive_mask = (
            differenced_series >= system_self_consumption)
        pct_over_zero = differenced_series_positive_mask.value_counts(
            normalize=True) * 100
        if pct_over_zero[True] >= pct_increase_threshold:
            return True
        else:
            return False


def cumulative_energy_avg_diff_check(energy_series,
                                     pct_increase_threshold=95,
                                     system_self_consumption=-0.5):
    """Check if an energy time series is cumulative via average differencing.

    To determine if an energy data stream is cumulative, subsequent values in
    the series are average differenced to determine if the data stream is
    consistently increasing. If the percentage of increasing values in the
    data set exceeds the pct_increase_threshold parameter, the energy series
    is determined as cumulative and a True boolean is returned.
    Otherwise, False is returned.

    Parameters
    ----------
    energy_series: Series
        Time series of energy data stream with datetime index.
    pct_increase_threshold: int, default 95
        The percentage threshold to consider the energy series as cumulative.
    system_self_consumption: Float, default -0.5
        The difference threshold to account for the effect of nighttime system
        self-consumption on the energy data stream.

    Returns
    -------
    Boolean
        True if energy_series is cumulative, False otherwise.
    """
    differenced_series = energy_series.diff().dropna()
    # Get the averaged difference
    avg_diff_series = 0.5 * \
        (differenced_series.shift(-1) + differenced_series).dropna()
    if len(differenced_series) == 0:
        warnings.warn(
            "The energy time series has a length of zero and cannot be run.")
        return False
    else:
        # If over X percent of the data is increasing (set via the
        # pct_increase_threshold), then assume that the column is cumulative
        avg_series_positive_mask = (avg_diff_series >= system_self_consumption)
        pct_over_zero = avg_series_positive_mask.value_counts(
            normalize=True) * 100
        if pct_over_zero[True] >= pct_increase_threshold:
            return True
        else:
            return False


def convert_cumulative_energy(energy_series, pct_increase_threshold=95,
                              system_self_consumption=-0.5):
    """Convert cumulative to interval-based, non-cumulative energy, if needed.

    Two main test are run to determine if the associated energy
    data stream is cumulative or not: a simple differencing function is run
    on the series via cumulative_energy_simple_diff_check, and an
    average differencing function is run on the series via
    cumulative_energy_avg_diff_check.

    Parameters
    ----------
    energy_series: Series
        Time series of energy data stream with datetime index.
    pct_increase_threshold: int, default 95
        The percentage threshold to consider the energy series as cumulative.
    system_self_consumption: Float, default -0.5
        The difference threshold to account for the effect of nighttime system
        self-consumption on the energy data stream.

    Returns
    -------
    Series
        corrected_energy_series is retuned if the energy series is cumulative.
        If the energy series passes the simple difference check, then the
        the series is corrected via the simple differencing. Else, if
        energy series passes the average difference check, then the series is
        corrected via average differencing.
        If neither checks are passes, then the original non-cumulative
        energy_series is returned.
    """
    # Check if energy series is cumulative with simple difference and average
    # difference
    simple_diff_check = cumulative_energy_simple_diff_check(
        energy_series, pct_increase_threshold, system_self_consumption)
    avg_diff_check = cumulative_energy_avg_diff_check(energy_series,
                                                      pct_increase_threshold,
                                                      system_self_consumption)
    if simple_diff_check:
        # Return simple difference of energy series if it passes the simple
        # difference check
        corrected_energy_series = energy_series.diff()
        return corrected_energy_series
    elif avg_diff_check:
        # Return average differnce of energy series if it passes the
        # average difference check
        corrected_energy_series = 0.5 * \
            (energy_series.diff().shift(-1) + energy_series.diff())
        return corrected_energy_series
    else:
        return energy_series
