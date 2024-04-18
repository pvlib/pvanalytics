"""
Quantifying the effects of snow cover
=====================================

We classify the effect of snow on a PV system's DC array. Snow on an array's
modules may reduce string voltage and/or string current, by reducing or
blocking irradiance from reaching the cells. These effects may vary across
the array since snow cover may not be uniform.

In this analysis, all differences between measured power and power modeled
from snow-free irradiance measurements are ascribed to the effects of snow. The
effect of snow is classified into one of five categories:

    * Mode 0: Indicates periods with enough opaque snow that the system is not
      producing power. Specifically, Mode 0 is when the measured voltage is
      below the inverter's turn-on voltage but the voltage modeled using
      measured irradiance is below the inverter's turn-on voltage.
    * Mode 1: Indicates periods when the system has non-uniform snow and
      both operating voltage and current are decreased. Operating voltage is
      reduced when bypass diodes activate and current is decreased due to
      decreased irradiance.
    * Mode 2: Indicates periods when the operating voltage is reduced but
      current is consistent with snow-free conditions.
    * Mode 3: Indicates periods when the operating voltage is consistent with
      snow-free conditionss but current is reduced.
    * Mode 4: Voltage and current are consistent with snow-free conditions.

    Mode is None when both measured and voltage modeled from measured
    irradiance are below the inverter turn-on voltage.

The procedure involves four steps:
    1. Using measured plane-of-array (POA) irradiance and temperature, model
       the module's maximum power current (Imp) and voltage (Vmp) assuming
       that all the POA irradiance reaches the module's cells.
    2. Use the modeled Imp and measured Imp, determine the fraction of
       plane-of-array irradiance that reaches the module's cells. This fraction
       is called the transmittance.
    3. Model the Vmp that would result from the POA irradiance reduced by
       the transmittance.
    4. Classify the effect of snow using the ratio of modeled Vmp (from step 3)
       and measured Vmp.
We demonstrate this analysis using measurements made at the combiner boxes
for a utility-scale system.

"""

# %% Import packages

import pathlib
import json
import pandas as pd
import numpy as np
import re
import pvlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import pvanalytics
# Functions needed for the analysis procedure
from pvanalytics.features import clipping
from pvanalytics.features import snow

# %% Load in system configuration parameters (dict)
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
data_file = pvanalytics_dir / 'data' / 'snow_data.csv'
snowfall_file = pvanalytics_dir / 'data' / 'snow_snowfall.csv'
horizon_file = pvanalytics_dir / 'data' / 'snow_horizon.csv'
config_file = pvanalytics_dir / 'data' / 'snow_config.json'

with open(config_file) as json_data:
    config = json.load(json_data)

# %% Retrieve and print system inverter specs and electrical configuration

cec_inverter_db = pvlib.pvsystem.retrieve_sam('CECInverter')
my_inverter = cec_inverter_db[config['inverter']]

max_ac_power = my_inverter['Paco']*0.001  # originally in W, convert to kW
mppt_low_voltage = my_inverter['Mppt_low']  # [V]
mppt_high_voltage = my_inverter['Mppt_high']  # [V]

print(f"Inverter AC power rating: {max_ac_power} kW")
print(f"Inverter MPPT range: {mppt_low_voltage} V - {mppt_high_voltage} V")
num_str_per_cb = config['num_str_per_cb']['INV1 CB1']
num_mods_per_str = config['num_mods_per_str']['INV1 CB1']
print(f"There are {num_str_per_cb} modules connected in series in each string,"
      f" and there are {num_mods_per_str} strings connected in"
      f" parallel at each combiner")


# %%
# Read in 15-minute sampled DC voltage and current time series data, AC power,
# module temperature collected by a BOM sensor and plane-of-array
# irradiance data collected by a heated pyranometer. This sample is provided
# by an electric utility.

# Load in utility data
data = pd.read_csv(data_file, index_col='Timestamp')
data.set_index(pd.DatetimeIndex(data.index, ambiguous='infer'), inplace=True)
data = data[~data.index.duplicated()]

# Explore utility datatset
print('Utility-scale dataset')
print('Start: {}'.format(data.index[0]))
print('End: {}'.format(data.index[-1]))
print('Frequency: {}'.format(data.index.inferred_freq))
print('Columns : ' + ', '.join(data.columns))
data.between_time('8:00', '16:00').head()

# Identify current, voltage, and AC power columns
dc_voltage_cols = [c for c in data.columns if 'Voltage' in c]
dc_current_cols = [c for c in data.columns if 'Current' in c]
ac_power_cols = [c for c in data.columns if 'AC' in c]

# Set negative or Nan current, voltage, AC power values to zero. This is
# allows us to calculate losses later.

data.loc[:, dc_voltage_cols] = np.maximum(data[dc_voltage_cols], 0)
data.loc[:, dc_current_cols] = np.maximum(data[dc_current_cols], 0)
data.loc[:, ac_power_cols] = np.maximum(data[ac_power_cols], 0)

data[dc_voltage_cols] = data[dc_voltage_cols].replace({np.nan: 0, None: 0})
data[dc_current_cols] = data[dc_current_cols].replace({np.nan: 0, None: 0})
data.loc[:, ac_power_cols] = data[ac_power_cols].replace({np.nan: 0, None: 0})

# %% Plot DC voltage for each combiner input relative to inverter nameplate
# limits
fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for v in dc_voltage_cols:
    ax.scatter(data.index, data[v], s=0.5, label=v)
    ax.plot(data[v], alpha=0.2)
ax.axhline(mppt_high_voltage, c='r', ls='--',
           label='Maximum MPPT voltage: {} V'.format(mppt_high_voltage))
ax.axhline(mppt_low_voltage, c='g', ls='--',
           label='Minimum MPPT voltage: {} V'.format(mppt_low_voltage))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('Voltage [V]', fontsize='large')
ax.legend(loc='lower left')

# %% Plot AC power relative to inverter nameplate limits

fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for a in ac_power_cols:
    ax.scatter(data.index, data[a], s=0.5, label=a)
    ax.plot(data[a], alpha=0.2)
ax.axhline(max_ac_power, c='r', ls='--',
           label='Maximum allowed AC power: {} kW'.format(max_ac_power))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('AC Power [kW]', fontsize='large')
ax.legend(loc='upper left')

# %% Filter data.
# Identify periods where the system is operating off of its maximum power
# point (MPP), and correct or mask. Conditions outside of the MPP cannot
# be accurately modeled without external information on the system's
# operating point. To allow us to make a valid comparison between system
# measurements and modeled power at MMP, we set measurements collected below
# the MPPT minimum voltage to zero, which emulates the condition where the
# inverter turns off when it cannot meet the turn-on voltage. When the inverter
# is clipping power, we replace voltage and current measurements with NaN as
# these measurements reflect current and voltage that has been artificially
# adjusted away from the MMP. This masking may result in an omission of some
# snow loss conditions where a very light-transmissive snow cover allows the
# system to reach the inverter's clipping voltage.

ac_power_cols_repeated = ac_power_cols + ac_power_cols + ac_power_cols
for v, i, a in zip(dc_voltage_cols, dc_current_cols, ac_power_cols_repeated):

    # Data where V > MPPT maximum
    data.loc[data[v] > mppt_high_voltage, v] = np.nan
    data.loc[data[v] > mppt_high_voltage, i] = np.nan
    data.loc[data[v] > mppt_high_voltage, a] = np.nan

    # Data where V < MPPT minimum
    data.loc[data[v] < mppt_low_voltage, v] = 0
    data.loc[data[v] < mppt_low_voltage, i] = 0
    data.loc[data[v] < mppt_low_voltage, a] = 0

    # Data where system is at Voc
    data.loc[data[i] == 0, v] = 0

    # Data where inverter is clipping based on AC power
    mask1 = data[a] > max_ac_power
    mask2 = clipping.geometric(ac_power=data[a], freq='15T')
    mask3 = np.logical_or(mask1.values, mask2.values)

    data.loc[mask3, v] = np.nan
    data.loc[mask3, i] = np.nan
    data.loc[mask3, a] = np.nan

# %% Plot DC voltage for each combiner input with inverter nameplate limits

fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for v in dc_voltage_cols:
    ax.scatter(data.index, data[v], s=0.5, label=v)
    ax.plot(data[v], alpha=0.2)
ax.axhline(mppt_high_voltage, c='r', ls='--',
           label='Maximum MPPT voltage: {} V'.format(mppt_high_voltage))
ax.axhline(mppt_low_voltage, c='g', ls='--',
           label='Minimum MPPT voltage: {} V'.format(mppt_low_voltage))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('Voltage [V]', fontsize='large')
ax.legend(loc='lower left')

# %% We want to exclude periods where array voltage is affected by horizon
# shading
'''
Load in and apply horizon profiling created using approach described in [1].

[1] J. L. Braid and B. G. Pierce, "Horizon Profiling Methods for Photovoltaic
Arrays," 2023 IEEE 50th Photovoltaic Specialists Conference (PVSC),
San Juan, PR, USA, 2023, pp. 1-7. doi:`10.1109/PVSC48320.2023.10359914`

'''

horizon = pd.read_csv(horizon_file, index_col='Unnamed: 0').squeeze("columns")

data['Horizon Mask'] = snow.get_horizon_mask(horizon, data['azimuth'],
                                             data['elevation'])

# %% Plot horizon mask

fig, ax = plt.subplots()
ax.scatter(data['azimuth'], data['elevation'], s=0.5, label='data',
           c=data['Horizon Mask'])
ax.scatter(horizon.index, horizon, s=0.5, label='mask')
ax.legend()
ax.set_xlabel(r'Azimuth [$\degree$]')
ax.set_ylabel(r'Elevation [$\degree$]')

# %% Exclude data collected while the sun is below the horizon
data = data[data['Horizon Mask']]

# %%

# Define coefficients for modeling transmission and voltage. User can either
# use the SAPM to calculate transmission or an approach based on the ratio
# between measured current and nameplate current. For modeling voltage, the
# user can use the SAPM or a single diode equivalent.

sapm_coeffs = config['sapm_coeff']
cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
sde_coeffs = cec_module_db[config['panel']]

# %%
"""
Model cell temperature using the SAPM model.
"""

irrad_ref = 1000
data['Cell Temp [C]'] = pvlib.temperature.sapm_cell_from_module(
    data['Module Temp [C]'], data['POA [W/m²]'], 3)

# %% Plot cell temperature
fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
ax.scatter(data.index, data['Cell Temp [C]'], s=0.5, c='b')
ax.plot(data['Cell Temp [C]'], alpha=0.3, c='b')
ax.set_ylabel('Cell Temp [C]', c='b', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='xx-large')

# %% For one combiner, demonstrate the transmission calculation using two
# different approaches to modeling effective irradiance from measured Imp.

# Choose one combiner box
j = 0

# Get key for configuration dict
matched = re.match(r'INV(\d+) CB(\d+)', dc_current_cols[j])
inv_cb = matched.group(0)

# Number of strings connected in parallel to combiner.
# Used to scale measured current down to the resolution
# of a single string connected in series, which should
# be the same current as a single module.

i_scaling_factor = int(config['num_str_per_cb'][f'{inv_cb}'])
# String current
imp = data[dc_current_cols[j]] / i_scaling_factor

# Approach 1 using SAPM
modeled_e_e1 = snow.get_irradiance_sapm(
    data['Cell Temp [C]'], imp, sapm_coeffs['Impo'], sapm_coeffs['C0'],
    sapm_coeffs['C1'], sapm_coeffs['Aimp'])

T1 = snow.get_transmission(data['POA [W/m²]'], modeled_e_e1, imp)

# Approach 2 using a linear irradiance-Imp model
modeled_e_e2 = snow.get_irradiance_imp(imp, sapm_coeffs['Impo'])

T2 = snow.get_transmission(data['POA [W/m²]'], modeled_e_e2, imp)

# %%
# Plot transmission calculated using two different approaches

fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d \n%H:%M")
ax.xaxis.set_major_formatter(date_form)

ax.scatter(T1.index, T1, s=0.5, c='b', label='SAPM')
ax.plot(T1, alpha=0.3, c='b')

ax.scatter(T2.index, T2, s=0.3, c='g', label='Linear model')
ax.plot(T2, alpha=0.3, c='g')

ax.legend()
ax.set_ylabel('Transmission', fontsize='xx-large')
ax.set_xlabel('Date + Time', fontsize='large')

# %%
# Model voltage using calculated transmission (two different approaches)

# Number of modules in series in a string
v_scaling_factor = int(config['num_mods_per_str'][inv_cb])

# Approach 1. Reduce measured POA using the transmission.
modeled_vmp_sapm = pvlib.pvsystem.sapm(data['POA [W/m²]']*T1,
                                       data['Cell Temp [C]'],
                                       sapm_coeffs)['v_mp']
modeled_vmp_sapm *= v_scaling_factor

# Approach 2  %TODO not sure we need this
# Code borrowed from pvlib-python/docs/examples/iv-modeling/plot_singlediode.py

# adjust the reference parameters according to the operating
# conditions using the De Soto model:
IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
    data['POA [W/m²]']*T1,
    data['Cell Temp [C]'],
    alpha_sc=sde_coeffs['alpha_sc'],
    a_ref=sde_coeffs['a_ref'],
    I_L_ref=sde_coeffs['I_L_ref'],
    I_o_ref=sde_coeffs['I_o_ref'],
    R_sh_ref=sde_coeffs['R_sh_ref'],
    R_s=sde_coeffs['R_s'],
   )

# plug the parameters into the SDE and solve for IV curves:
SDE_params = {
    'photocurrent': IL,
    'saturation_current': I0,
    'resistance_series': Rs,
    'resistance_shunt': Rsh,
    'nNsVth': nNsVth
}
modeled_vmp_sde = pvlib.pvsystem.singlediode(**SDE_params)['v_mp']
modeled_vmp_sde *= v_scaling_factor

# %% Plot modeled and measured voltage

fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form)

ax.scatter(modeled_vmp_sapm.index, modeled_vmp_sapm, s=1, c='b', label='SAPM')
ax.scatter(modeled_vmp_sde.index, modeled_vmp_sde, s=1, c='g', label='SDE')

ax.scatter(data.index, data[inv_cb + ' Voltage [V]'], s=1, c='r',
           label='Measured')
ax.legend(fontsize='xx-large')
ax.set_ylabel('Voltage [V]', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='large')


# %% Function to do analysis so we can loop over combiner boxes

def wrapper(voltage, current, temp_cell, effective_irradiance,
            coeffs, config, temp_ref=25, irrad_ref=1000):

    '''
    Categorizes each data point as Mode 0-4 based on transmission and
    the ratio between measured and modeled votlage.

    This function illustrates a workflow to get to snow mode:

    1. Model effective irradiance based on measured current using the SAPM.
    2. Calculate transmission.
    3. Model voltage from measured irradiance reduced by transmission. Assume
       that all strings are producing voltage.
    4. Determine the snow mode for each point in time.

    Parameters
    ----------
    voltage : array
        Voltage [V] measured at inverter.
    current : array
        Current [A] measured at combiner.
    temp_cell : array
        Cell temperature. [degrees C]
    effective_irradiance : array
        Snow-free POA irradiance measured by a heated pyranometer. [W/m²]
    coeffs : dict
        A dict defining the SAPM parameters, used for pvlib.pvsystem.sapm.
    config : dict
        min_dcv : float
            The lower voltage bound on the inverter's maximum power point
            tracking (MPPT) algorithm. [V]
        max_dcv : numeric
            Upper bound voltage for the MPPT algorithm. [V]
        threshold_vratio : float
            The lower bound on vratio that is found under snow-free conditions,
            determined empirically.
        threshold_transmission : float
            The lower bound on transmission that is found under snow-free
            conditions, determined empirically.
        num_mods_per_str : int
            Number of modules in series in each string.
        num_str_per_cb : int
            Number of strings in parallel at the combiner.

    Returns
    -------
    my_dict : dict
        Keys are ``'transmission'``, ``'modeled_vmp'``, ``'vmp_ratio'``,
        and ``'mode'``

    '''

    # Calculate transmission
    modeled_e_e = snow.get_irradiance_sapm(
        temp_cell, current/config['num_str_per_cb'], coeffs['Impo'],
        coeffs['C0'], coeffs['C1'], coeffs['Aimp'])

    T = snow.get_transmission(effective_irradiance, modeled_e_e,
                              current/config['num_str_per_cb'])

    name_T = inv_cb + ' Transmission'
    data[name_T] = T

    # Model voltage for a single module, scale up to array
    modeled_vmp = pvlib.pvsystem.sapm(effective_irradiance*T, temp_cell,
                                      coeffs)['v_mp']
    modeled_vmp *= config['num_mods_per_str']

    # Voltage is modeled as NaN if T = 0, but V = 0 makes more sense
    modeled_vmp[T == 0] = 0

    # Identify periods where modeled voltage is outside of MPPT range,
    # and correct values
    modeled_vmp[modeled_vmp > config['max_dcv']] = np.nan
    modeled_vmp[modeled_vmp < config['min_dcv']] = 0

    # Calculate voltage ratio
    with np.errstate(divide='ignore'):
        vmp_ratio = voltage / modeled_vmp

    # take care of divide by zero
    vmp_ratio[modeled_vmp == 0] = np.nan

    mode = snow.categorize(vmp_ratio, T, voltage, modeled_vmp,
                           config['min_dcv'],
                           config['threshold_vratio'],
                           config['threshold_transmission'])
    my_dict = {'transmission': T,
               'modeled_vmp': modeled_vmp,
               'vmp_ratio': vmp_ratio,
               'mode': mode}

    return my_dict


# %%
# Demonstrate transmission, modeled voltage calculation and mode categorization
# on voltage, current pair

j = 0
v = dc_voltage_cols[j]
i = dc_current_cols[j]

# Used to get key for configuration dict
matched = re.match(r'INV(\d+) CB(\d+)', i)
inv_cb = matched.group(0)

# Number of strings connected in parallel at the combiner
i_scaling_factor = int(config['num_str_per_cb'][f'{inv_cb}'])

# threshold_vratio and threshold-transmission were empirically determined
# using data collected on this system over five summers. Transmission and
# vratio were calculated for all data collected in the summer (under the
# assumption that there was no snow present at this time), and histograms
# of the spread of transmission and vratio values were made.
# threshold_vratio is the lower bound on the 95th percentile of all vratios
# calculated for summer data - as in, 95% of all data collected in the summer
# has a vratio that is higher than threshold_vratio. threshold_transmission
# is the same value, but for transmission calculated from data recorded
# during the summer.

threshold_vratio = config['threshold_vratio']
threshold_transmission = config['threshold_transmission']

my_config = {'threshold_vratio': threshold_vratio,
             'threshold_transmission': threshold_transmission,
             'min_dcv': mppt_low_voltage,
             'max_dcv': mppt_high_voltage,
             'num_str_per_cb': int(config['num_str_per_cb'][f'{inv_cb}']),
             'num_mods_per_str': int(config['num_mods_per_str'][f'{inv_cb}'])}

out = wrapper(data[v], data[i],
              data['Cell Temp [C]'],
              data['POA [W/m²]'], sapm_coeffs,
              my_config)

# %%
# Calculate the transmission, model the voltage, and categorize into modes for
# all combiners. Use the SAPM to calculate transmission and model the voltage.

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):

    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}'

    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])
    i_scaling_factor = int(
        config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])

    my_config = {
        'threshold_vratio': threshold_vratio,
        'threshold_transmission': threshold_transmission,
        'min_dcv': mppt_low_voltage,
        'max_dcv': mppt_high_voltage,
        'num_str_per_cb': int(config['num_str_per_cb'][f'{inv_cb}']),
        'num_mods_per_str': int(config['num_mods_per_str'][f'{inv_cb}'])}

    out = wrapper(data[v_col], data[i_col],
                  data['Cell Temp [C]'],
                  data['POA [W/m²]'], sapm_coeffs,
                  my_config)

    for k, v in out.items():
        data[inv_cb + ' ' + k] = v


# %%
# Look at transmission for all DC inputs

transmission_cols = [c for c in data.columns if 'transmission' in c]
fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in transmission_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)
ax.set_xlabel('Date', fontsize='xx-large')
ax.legend()

# %%
# Look at voltage ratios for all DC inputs

vratio_cols = [c for c in data.columns if "vmp_ratio" in c]
fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in vratio_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('Voltage Ratio (measured/modeled)', fontsize='xx-large')
ax.axhline(1, c='k', alpha=0.1, ls='--')
ax.legend()

# %% Calculate all power losses - snow and non-snow

modeled_df = pvlib.pvsystem.sapm(data['POA [W/m²]'],
                                 data['Cell Temp [C]'],
                                 sapm_coeffs)

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):
    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}'
    i_scaling_factor = int(
        config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])
    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])

    modeled_power = modeled_df['p_mp']*v_scaling_factor*i_scaling_factor
    name_modeled_power = inv_cb + ' Modeled Power [W]'
    data[name_modeled_power] = modeled_power

    name_loss = inv_cb + ' Loss [W]'
    loss = np.maximum(data[name_modeled_power] - data[i_col]*data[v_col], 0)
    data[name_loss] = loss

# %%

loss_cols = [c for c in data.columns if "Loss" in c]
mode_cols = [c for c in data.columns if "mode" in c and "modeled" not in c]
modeled_power_cols = [c for c in data.columns if "Modeled Power" in c]

col = 1
los = loss_cols[col]
mod = mode_cols[col]
pwr = modeled_power_cols[col]

# Color intervals by mode
cmap = {0: 'r',
        1: 'b',
        2: 'yellow',
        3: 'cyan',
        4: 'g'}

fig, ax = plt.subplots(figsize=(10, 10))
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data[~data[mod].isna()]

# Plot each day individually so we are not exaggerating losses
days_mapped = temp.index.map(lambda x: x.date())
days = np.unique(days_mapped)
grouped = temp.groupby(days_mapped)

for d in days:
    temp_grouped = grouped.get_group(d)
    ax.plot(temp_grouped[pwr], c='k', ls='--')
    ax.plot(temp_grouped[pwr] - temp_grouped[los], c='k')
    ax.fill_between(temp_grouped.index, temp_grouped[pwr] - temp_grouped[los],
                    temp_grouped[pwr], color='k', alpha=0.2)

    chng_pts = np.ravel(np.where(temp_grouped[mod].values[:-1]
                                 - temp_grouped[mod].values[1:] != 0))

    if len(chng_pts) == 0:
        ax.axvspan(temp_grouped.index[0], temp_grouped.index[-1],
                   color=cmap[temp_grouped.at[temp_grouped.index[-1], mod]],
                   alpha=0.05)
    else:
        set1 = np.append([0], chng_pts)
        set2 = np.append(chng_pts, [-1])

        for start, end in zip(set1, set2):
            my_index = temp_grouped.index[start:end]
            ax.axvspan(
                temp_grouped.index[start], temp_grouped.index[end],
                color=cmap[temp_grouped.at[temp_grouped.index[end], mod]],
                alpha=0.05)

# Add different colored intervals to legend
handles, labels = ax.get_legend_handles_labels()

modeled_line = Line2D([0], [0], label='Modeled', color='k', ls='--')
measured_line = Line2D([0], [0], label='Measured', color='k')

# TODO don't mark offline periods as Mode 0
red_patch = mpatches.Patch(color='r', alpha=0.05, label='Mode 0')
blue_patch = mpatches.Patch(color='b', alpha=0.05, label='Mode 1')
yellow_patch = mpatches.Patch(color='y', alpha=0.05, label='Mode 2')
purple_patch = mpatches.Patch(color='cyan', alpha=0.05, label='Mode 3')
green_patch = mpatches.Patch(color='g', alpha=0.05, label='Mode 4')

handles.append(measured_line)
handles.append(modeled_line)
handles.append(red_patch)
handles.append(green_patch)
handles.append(blue_patch)
handles.append(yellow_patch)
handles.append(purple_patch)
# handles.append(gray_patch)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('DC Power [W]', fontsize='xx-large')
ax.legend(handles=handles, fontsize='xx-large', loc='upper left')
ax.set_title('Measured and modeled production for INV1 CB2',
             fontsize='xx-large')

# %%

# Calculate daily losses occuring while the system is operating in one of the
# four modes that are associated with the presence of snow.

# Daily snowfall is measured at 7:00 am of each day
snow = pd.read_csv(snowfall_file, index_col='DATE')  # originally in [mm]
snow['SNOW'] *= 1/(10*2.54)  # convert to [in]

loss_cols = [c for c in data.columns if "Loss" in c]
mode_cols = [c for c in data.columns if "mode" in c and "modeled" not in c]
modeled_power_cols = [c for c in data.columns if "Modeled Power" in c]

days_mapped = data.index.map(lambda x: x.date())
days = np.unique(days_mapped)
data_gped = data.groupby(days_mapped)

columns = [re.match(r'INV(\d) CB(\d)', c).group(0) for c in loss_cols]

snow_loss = pd.DataFrame(index=days, columns=columns)

for d in days:
    temp = data_gped.get_group(d)

    for c, m, l, p in zip(columns, mode_cols, loss_cols, modeled_power_cols):
        snow_loss_filter = ~(temp[m].isna()) & (temp[m] != 4)
        daily_snow_loss = 100*temp[snow_loss_filter][l].sum()/temp[p].sum()
        snow_loss.at[d, c] = daily_snow_loss


fig, ax = plt.subplots()
date_form = DateFormatter("%m/%d")

days_mapped = data.index.map(lambda x: x.date())
days = np.unique(days_mapped)

xvals = np.arange(0, len(days), 1)
xwidth = 0.05
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, c in enumerate(columns):
    ax.bar(xvals + xwidth*i, snow_loss[c], width=xwidth, color=colors[i],
           ec='k', label=c)

ax.legend()
ax.set_ylabel('[%]', fontsize='xx-large')
ax.set_xticks(xvals, days)
ax.xaxis.set_major_formatter(date_form)
ax.set_title('Losses incurred in modes 0 -3', fontsize='xx-large')
