"""
Clipping Detection
===================

Identifying clipping periods using the PVAnalytics clipping module.
"""

# %%
# Identifying and removing clipping periods from AC power time series
# data aids in generating more accurate degradation analysis results,
# as using clipped data can lead to over-predicting degradation. In this
# example, we show how to use
# :py:func:`pvanalytics.features.clipping.geometric`
# to mask clipping periods in an AC power time series. We use a
# normalized time series example provided by the PV Fleets Initiative,
# where clipping periods are labeled as True, and non-clipping periods are
# labeled as False. This example is provided taken from the DuraMAT DataHub:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data

import pvanalytics
from pvanalytics.features.clipping import geometric
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np

# %%
# First, read in the ac_power_inv_7539 example, and visualize a subset of the
# clipping periods via the "label" mask column.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = pvanalytics_dir / 'data' / 'ac_power_inv_7539.csv'
data = pd.read_csv(ac_power_file_1, index_col=3, parse_dates=True)
data['label'] = data['label'].astype(bool)
freq = str(int(data.index.to_series().diff().value_counts().idxmax()
               .seconds/60))+"T"
scatter = plt.scatter(data.index[3500:4000],
                      data['value_normalized'][3500:4000],
                      c=data.label[3500:4000].astype('int'),
                      cmap="bwr")
plt.legend(handles=scatter.legend_elements()[0],
           labels=[False, True],
           title="Clipped")
plt.xticks(rotation=45)
plt.xlabel("Date", size=18)
plt.ylabel("Normalized AC Power", size=18)
plt.tight_layout()
plt.show()

# %%
# Now, use :py:func:`pvanalytics.features.clipping.geometric` to identify
# clipping periods in the time series. Re-plot the data subset with this mask.
predicted_clipping_mask = geometric(ac_power=data['value_normalized'],
                                    freq=freq)
scatter = plt.scatter(data.index[3500:4000],
                      data['value_normalized'][3500:4000],
                      c=predicted_clipping_mask[3500:4000].astype('int'),
                      cmap="bwr")
plt.legend(handles=scatter.legend_elements()[0],
           labels=[False, True],
           title="Clipped")
plt.xticks(rotation=45)
plt.xlabel("Date", size=18)
plt.ylabel("Normalized AC Power", size=18)
plt.tight_layout()
plt.show()


# %%
# Compare the filter results to the ground-truth labeled data side-by-side,
# and generate an accuracy metric.
acc = 100 * np.sum(np.equal(data.label,
                            predicted_clipping_mask))/len(data.label)
print("Overall model prediction accuracy: " + str(round(acc, 2)) + "%")
