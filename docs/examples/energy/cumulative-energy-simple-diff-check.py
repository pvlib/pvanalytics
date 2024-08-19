"""
Cumulative Energy via Simple Difference
=======================================

Check and correct energy series for cumulative energy with simple differencing.
"""

# %%
# AC energy data streams are often cumulative, meaning they increase
# over time. These energy streams need to be corrected into a form
# that resembles regular, non-cumulative data. This process involves
# applying a percent threshold to the difference in subsequent values
# to check if the data is always increasing. If it passes,
# that check, then the data stream is cumulative and it can be corrected
# via simple differencing.

import pandas as pd
import matplotlib.pyplot as plt
from pvanalytics.quality import energy
import pathlib
import pvanalytics

# %%
# First, read in the ac energy data. This data set contains one week of
# 10-minute ac energy data.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
energy_file = pvanalytics_dir / 'data' / 'system_10004_ac_energy.csv'
data = pd.read_csv(energy_file)
energy_series = data['ac_energy_inv_16425']

# %%
# Now check if the energy time series is cumulative via simple differencing.
# This is done using the
# :py:func:`pvanalytics.quality.energy.cumulative_energy_simple_diff_check`
# function.

is_cumulative = energy.cumulative_energy_simple_diff_check(energy_series)

# %%
# If the energy series is cumulative, then it can be converted
# to the non-cumulative energy series. This is done using the
# :py:func:`pvanalytics.quality.energy.convert_cumulative_energy` function.
corrected_energy_series = energy.convert_cumulative_energy(energy_series)

# %%
# Plot the original, cumulative energy series.

data.plot(x="local_measured_on", y='ac_energy_inv_16425')
plt.title("Cumulative Energy Series")
plt.xticks(rotation=45)
plt.xlabel("Datetime")
plt.ylabel("AC Energy (kWh)")
plt.tight_layout()
plt.show()

# %%
# Plot the corrected, non-cumulative energy series.

corrected_energy_df = pd.DataFrame({
    "local_measured_on": data["local_measured_on"],
    "corrected_ac_energy_inv_16425": corrected_energy_series})
corrected_energy_df = corrected_energy_df[
    corrected_energy_df["corrected_ac_energy_inv_16425"] >= 0]
corrected_energy_df.plot(x="local_measured_on",
                         y="corrected_ac_energy_inv_16425")
plt.title("Corrected, Non-cumulative Energy Series")
plt.xticks(rotation=45)
plt.xlabel("Datetime")
plt.ylabel("AC Energy (kWh)")
plt.tight_layout()
plt.show()
