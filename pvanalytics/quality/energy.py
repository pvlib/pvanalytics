"""Quality control functions for energy data."""
import warnings


def cumulative_energy_simple_diff_check(energy_series,
                                        pct_increase_threshold=95,
                                        system_self_consumption=-0.5):
    """Check if an energy time series has cumulative energy or not.

    The check uses the simple diff function .diff().

    Parameters
    ----------
    energy_series : Pandas series with datetime index.
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
    """Check if an energy time series has cumulative energy or not.

    The check uses the average difference.

    Parameters
    ----------
    energy_series : Pandas series with datetime index.
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


def check_cumulative_energy(energy_series, pct_increase_threshold=95,
                            system_self_consumption=-0.5):
    """Run the cumulative energy check for simple and averaged difference.

    Parameters
    ----------
    energy_series : Pandas series with datetime index.
        Time series of energy data stream with datetime index.
    pct_increase_threshold: int, default 95
        The percentage threshold to consider the energy series as cumulative.
    system_self_consumption: Float, default -0.5
        The difference threshold to account for the effect of nighttime system
        self-consumption on the energy data stream.

    Returns
    -------
    Tuple
        (simple_diff_energy_series, avg_diff_energy_series, cumulative_energy)

        simple_diff_energy_series: Pandas series with datetime index.
            The differenced energy series using the simple .diff() function if
            energy is cumulative; otherwise, it remains as the original,
            noncumulative energy series.
        avg_diff_energy_series: Pandas series with datetime index.
            The averaged difference energy series using the averaged difference
            method if energy is cumulative; otherwise, it remains as the
            original, noncumulative energy series.
        cumulative_energy: Boolean
            True if energy series is cumulative, False otherwise.
    """
    # Check if energy series is cumulative for both simple and averaged diff
    cumulative_energy = (
        cumulative_energy_simple_diff_check(energy_series,
                                            pct_increase_threshold,
                                            system_self_consumption) and
        cumulative_energy_avg_diff_check(energy_series,
                                         pct_increase_threshold,
                                         system_self_consumption))
    if cumulative_energy:
        # Adjust energy series if it is cumulative
        simple_diff_energy_series = energy_series.diff()
        avg_diff_energy_series = 0.5 * \
            (simple_diff_energy_series.shift(-1) + simple_diff_energy_series)
        return (simple_diff_energy_series, avg_diff_energy_series,
                cumulative_energy)
    else:
        simple_diff_energy_series = energy_series
        avg_diff_energy_series = energy_series
        return (simple_diff_energy_series, avg_diff_energy_series,
                cumulative_energy)
