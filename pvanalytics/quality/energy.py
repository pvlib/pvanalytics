"""Quality control functions for energy data."""
import warnings


def is_cumulative_energy(energy_series, pct_increase_threshold=95):
    """Check if an energy time series represents cumulative energy or not.

    Parameters
    ----------
    energy_series : series
        Series of energy data.
    pct_increase_threshold: int, default 95
        The percentage threshold to consider the energy series as cumulative.

    Returns
    -------
    Boolean
        True if energy_series is cumulative, False otherwise.
    """
    differenced_series = energy_series.diff()
    differenced_series = differenced_series.dropna()
    if len(differenced_series) == 0:
        warnings.warn(
            "The energy time series has a length of zero and cannot be run.")
        return False
    else:
        # If over X percent of the data is increasing (set via the
        # pct_increase_threshold), then assume that the column is cumulative
        differenced_series_positive_mask = (differenced_series >= -.5)
        pct_over_zero = differenced_series_positive_mask.value_counts(
            normalize=True) * 100
        if pct_over_zero[True] >= pct_increase_threshold:
            return True
        else:
            return False


def check_cumulative_energy(energy_series, pct_increase_threshold=95):
    """Run the cumulative energy check and adjust the energy series.

    Parameters
    ----------
    energy_series : series
        Series of energy data.
    pct_increase_threshold: int, default 95
        The percentage threshold to consider the energy series as cumulative.

    Returns
    -------
    Tuple
        (energy_series, cumulative_energy)
        Energy_series is the differenced series if cumulative and
        cumulative_energy is a boolean to indicate if the series is
        cumulative.
    """
    # Check if energy series is cumulative
    cumulative_energy = is_cumulative_energy(
        energy_series, pct_increase_threshold)
    if cumulative_energy:
        # Adjust energy series if it is cumulative
        energy_series = energy_series.diff()
    return energy_series, cumulative_energy
