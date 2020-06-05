"""Functions for identifying when the sun is shining."""
from pvanalytics.util import _fit


def sunny_days(power_or_irradiance, daytime, correlation_min=0.94,
               fixed_max=0.96, tracking=False):
    """Return True for values on days that are sunny.

    Sunny days are identified when the :math:`r^2` for a quadratic fit
    to the power data is greater than `correlation_min`. If `tracking`
    is True then a quartic is fit instead of a quadratic.

    Parameters
    ----------
    power_or_irradiance : Series
        Timezone localized series of power or irradiance measurements.
    daytime : Series
        Boolean series with True for times that are during the
        day. For best results this mask should exclude early morning
        and evening as well as night. Morning and evening may have
        problems with shadows that interfere with curve fitting, but
        do not necessarily indicate that the day was not sunny.
    correlation_min : float, default 0.94
        Minimum :math:`r^2` for a day to be considered sunny.
    fixed_max : float, default 0.96
        Maximum :math:`r^2` for a quadratic fit when `tracking=True`
    tracking : bool, default False
        Whether the system has a tracker.

    Notes
    -----
    Based on the PVFleets QA Analysis project. Copyright (c) 2020
    Alliance for Sustainable Energy, LLC.

    """
    # PVFleets has two funcitons: sunny_days() which is used for fixed
    # systems, and tracking_onsun_days() which is used for tracking
    # systems. sunny_days() just fits the quadratic, but
    # tracking_onsun_days() fits a quartic _and_ a quadratic, if the
    # r^2 for the quadratic fit is too large (> 0.96) or is larger
    # than the r^2 for the quartic fit then the system is not marked
    # as 'sunny'/'tracking'.
    #
    # TODO reconcile the differences between the fixed/tracking
    # functions:
    #
    #   | fixed                    | tracking                 |
    #   |--------------------------+--------------------------|
    #   | remove data less than    | remove data less than    |
    #   | 0.4*mean(positive data)  | 0.1*mean(positive data)  |
    #   |--------------------------+--------------------------|
    #   | N/A                      | only fit curves if max   |
    #   |                          | of day is greater than   |
    #   |                          | > mean(positive data)    |
    #   |--------------------------+--------------------------|
    #   | only fit curves if there | only fit curves if there |
    #   | are at least 5 hours of  | are at least 5 hours of  |
    #   | data on a day            | data on a day            |
    #
    # In all these cases, if the condition is not satisfied then r^2
    # for that day is set to 0.0 which will cause the day to be marked
    # False.
    #
    # TODO tracking_onsun_check() severely limits the amount of the
    # day for the quadratic fit. From the comments:
    #
    #     the following line limits the minutes of day for the
    #     quadratic the intent is to remove early morning and late
    #     afternoon hours that can skew results because a stuck
    #     tracker often will have a morning or afternoon tail
    #     depending on stuck orientation
    #     >>> data = data[530:910]
    #
    # NOTE should use data.between_time('08:45', '15:15') here, since
    # we have a DatetimeIndex still
    daytime_data = power_or_irradiance[daytime].resample('D')
    sunny_fixed = daytime_data.apply(_fit.quadratic)
    if tracking:
        sunny_tracking = daytime_data.apply(_fit.quartic)
        tracking_days = (
            (sunny_tracking > correlation_min)
            & (sunny_fixed < fixed_max)
            & (sunny_fixed < sunny_tracking)
        )
        # TODO Make sure that when we reindex, Trues don't propagate past
        # midnight if there is a missing day following a sunny day.
        return tracking_days.reindex(power_or_irradiance.index, method='pad')
    return (sunny_fixed > correlation_min).reindex(power_or_irradiance.index, method='pad')
