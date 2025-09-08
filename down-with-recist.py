'''
Algorithm:
1. Artificially generate a dataset of N patients (input arg: N).
2. For each patient, generate a random number of lesions (range: 1-30, poisson distribution with µ = 5).
3. For each lesion, generate a random diameter (select from a list of possible diameters, with replacement (using original_shape_Maximum2DDiameterSlice)).
4. For each lesion, generate a diameter change (select from a Gaussian distribution, with fractional changes from -1 to 1).
5. For each lesion, assign a location tag (e.g. liver, lung, bone, brain, again from a list of possible locations, with replacement).
----- LESION SELECTION BIAS ASSESSMENT -----
6. For each patient, select a random number of lesions to be target lesions (up to 5 per patient, with a maximum of 2 per assigned location).
7. From the selected target lesions, assess the RECIST response (SLD = 100% decrease = CR, SLD > 30% decrease = PR, SLD > 20% increase = PD, otherwise SD).
8. For each patient, assess the RECIST response across all lesions (as above).
9. Compare the RECIST response across all lesions to the RECIST response across target lesions only.
----- DIAMETER VERSUS VOLUME ASSESSMENT -----
10. For each diameter, calculate the corresponding volume (assuming spherical shape; this will represent the lower end of volumetric measurement).
11. Inject inter-observer variability by adding a random error to the volume calculation (select from a Gaussian distribution, with fractional diameter changes from -0.113 to 0.113; REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC3423763/).
11. For each lesion, calculate the volume using another diameter measurement (i.e., original_shape_Maximum3DDiameter).


'''
# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
random.seed(165)

# Sample data
rad_data = pd.read_csv('data/radiomics-anon.csv')
rad_data['slice_thickness'] = [float(i.split(', ')[-1].strip('()')) for i in rad_data['diagnostics_Image-interpolated_Spacing']]
rad_data['volume (cc); contoured'] = rad_data['original_shape_VoxelVolume'] * rad_data['slice_thickness'] / 1000

# Config
N = 10000  # Number of patients
mu = 10  # Expected number of lesions per patient

# Truncated normal distribution - diameter change
a, b = -1, 3  # Min and max values
mean, std_dev = 0, 0.3  # Mean and standard deviation
trunc_normal_dist = truncnorm((a - mean) / std_dev, (b - mean) / std_dev, loc=mean, scale=std_dev)

# %% Generate data (lesion level)
data = []
for i in range(N):
    # Generate random number of lesions with poisson distribution (min 1 and max 30)
    n_lesions = np.random.poisson(mu)
    while n_lesions < 1 or n_lesions > 30:
        n_lesions = np.random.poisson(mu)
    for j in range(n_lesions):
        # take a random index
        ind = random.choice(rad_data.index)
        # Generate diameter (pre-treatment)
        diameter_pre = rad_data['original_shape_Maximum2DDiameterSlice'][ind]
        # Generate diameter change
        diameter_change = trunc_normal_dist.rvs(size=1)[0]
        # Generate diameter (post-treatment)
        diameter_post = diameter_pre + diameter_pre * diameter_change
        # Generate location tag
        location = rad_data['LABEL'][ind]
        # Volume (contoured)
        volume = rad_data['volume (cc); contoured'][ind]
        # Other data
        diameter_3Dmax = rad_data['original_shape_Maximum3DDiameter'][ind]
        diameter_majorAx = rad_data['original_shape_MajorAxisLength'][ind]
        diameter_minorAx = rad_data['original_shape_MinorAxisLength'][ind]

        # Append to data
        data.append([i, j, diameter_pre, diameter_change, diameter_post, location, volume, diameter_3Dmax, diameter_majorAx, diameter_minorAx])

# Convert to DataFrame
synth_lesions = pd.DataFrame(data, columns=['patient', 'lesion', 'diameter-pre', 'diameter_change', 'diameter-post', 'location', 'volume (cc); contoured', 'diameter_3Dmax', 'diameter_majorAx', 'diameter_minorAx'])
synth_lesions['patient'] = synth_lesions['patient'].astype(str)
patients, counts = np.unique(synth_lesions['patient'], return_counts=True)


# %% Generate data (patient level)

SLD_pre = [np.sum(synth_lesions['diameter-pre'][synth_lesions['patient'] == p]) for p in patients]
SLD_post = [np.sum(synth_lesions['diameter-post'][synth_lesions['patient'] == p]) for p in patients]
SLD_chg = [(SLD_post[i] - SLD_pre[i])/SLD_pre[i] * 100 for i in range(len(SLD_pre))]


def recist_assess(SLD_chg):
    
    # Initialize RECIST response vector
    RECIST_response = []

    for i, patient in enumerate(patients):
        if SLD_chg[i] > 20:
            RECIST_response.append('PD')
        elif -30 < SLD_chg[i] <= 20:
            RECIST_response.append('SD')
        elif -100 < SLD_chg[i] <= -30:
            RECIST_response.append('PR')
        elif SLD_chg[i] == -100:
            RECIST_response.append('CR')

    return RECIST_response



# Add RECIST response to DataFrame
synth_patients = pd.DataFrame({'patient': np.sort(np.unique(synth_lesions.patient).astype(int)), 'num_lesions': counts, 'SLD_pre': SLD_pre, 'SLD_post': SLD_post, 'SLD_chg': SLD_chg, 'RECIST (all)': recist_assess(SLD_chg)})

# %% select target lesions

def target_selection(lesions_to_select):

    # ensure that there is not more than 2 target lesions per location
    target_lesions = []
    for patient in patients:

        # get the locations for the patient
        locations = np.unique(synth_lesions['location'][synth_lesions['patient'] == patient])
        patient_inds = []

        # for each location, select up to 2 lesions
        for location in locations:
            # get the indices of the lesions at the location
            indices = synth_lesions[(synth_lesions['patient'] == patient) & (synth_lesions['location'] == location)].index
            # select up to 2 lesions
            if len(indices) > 2:
                # select 2 random indices
                indices = random.sample(list(indices), 2)
                patient_inds.extend(indices)
            else:
                patient_inds.extend(indices)

        if len(patient_inds) > lesions_to_select:
            target_lesions.extend(random.sample(patient_inds, lesions_to_select))
        else:
            target_lesions.extend(patient_inds)

    return target_lesions


# choose 1-10 target lesions per patient (max 2 per location) - random selection
for i in range(1, 11):
    
    # generate the 'selected' dataset
    synth_lesions_select = synth_lesions.copy().iloc[target_selection(i)]
    # put in ascending order by index
    synth_lesions_select.sort_index(inplace=True)

    # evaluate the SLD_chg for the selected lesions
    SLD_pre_select = [np.sum(synth_lesions_select['diameter-pre'][synth_lesions_select['patient'] == p]) for p in patients]
    SLD_post_select = [np.sum(synth_lesions_select['diameter-post'][synth_lesions_select['patient'] == p]) for p in patients]
    SLD_chg_select = [(SLD_post_select[i] - SLD_pre_select[i])/SLD_pre_select[i] * 100 for i in range(len(SLD_pre_select))]

    recist = recist_assess(SLD_chg_select)
    synth_patients['RECIST (target) ' + str(i)] = recist


# %%

plt.rcParams.update({'font.size': 16})

# calculate the classification accuracy for RECIST as a function of the number of target lesions
acc = []
for i in range(1, 11):

    inds = synth_patients['num_lesions']>i
    acc.append(np.sum(synth_patients['RECIST (all)'][inds] == synth_patients['RECIST (target) ' + str(i)][inds]) / np.sum(inds) * 100)
plt.figure(figsize=(8, 6))
plt.axvline(5, color='red', linestyle='--', label='RECIST v1.1')
plt.scatter(range(1, 11), 1 - np.array(acc) / 100, marker='o', s=75, label = 'Observation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.xlabel('Target Lesions')
plt.ylabel('Misclassified Patients')
plt.show()

# show plot of sensitivity for detecting PD

sensitivity = []
for i in range(1, 11):
    inds = np.logical_and(synth_patients['num_lesions']>i, synth_patients['RECIST (all)'] == 'PD')
    sensitivity.append(np.sum(synth_patients['RECIST (target) ' + str(i)][inds] == 'PD') / np.sum(inds) * 100)
plt.figure(figsize=(8, 6))
plt.axvline(5, color='red', linestyle='--', label='RECIST v1.1')
plt.scatter(range(1, 11), np.array(sensitivity) / 100, marker='o', s=75, label = 'Observation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.xlabel('Target Lesions')
plt.ylabel('Sensitivity')
plt.show()

# %%
# Plot sensitivity and misclassification rate on the same plot
fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:blue'
ax1.set_xlabel('Target Lesions')
ax1.set_ylabel('Misclassified Patients', color=color)
ax1.scatter(range(1, 11), 1 - np.array(acc) / 100, marker='o', s=75, label='Misclassified Patients', color=color)
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:brown'
ax2.set_ylabel('Sensitivity', color=color)  # we already handled the x-label with ax1
ax2.scatter(range(1, 11), np.array(sensitivity) / 100, marker='o', s=75, label='Sensitivity', color=color)
ax2.tick_params(axis='y', labelcolor=color, length=0)
ax1.axvline(5, color='red', linestyle='--', label='RECIST v1.1')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Combine handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

# Place legend outside, 1 row x 3 columns
fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
sns.despine()
plt.show()


# %%

# using the diameter to calculate the volume
def volume_calc(diameter):
    return (4/3 * np.pi * (diameter/2)**3) / 1000

# calculate the volume for each lesion
synth_lesions['volume (cc); pre'] = volume_calc(synth_lesions['diameter-pre'])
synth_lesions['volume (cc); post'] = volume_calc(synth_lesions['diameter-post'])
synth_lesions['volume (cc); 3Dmax'] = volume_calc(synth_lesions['diameter_3Dmax'])
synth_lesions['volume (cc); majorAx'] = volume_calc(synth_lesions['diameter_majorAx'])
synth_lesions['volume (cc); minorAx'] = volume_calc(synth_lesions['diameter_minorAx'])



# %%

oct = pd.read_csv('data/radiomics-anon2.csv')

# Find pairs in the csv where 'USUBJID' and 'RoiNumber' are the same
oct['study_date'] = pd.to_datetime(oct['study_date'], format='%Y%m%d')
oct.sort_values(by=['USUBJID', 'RoiNumber', 'study_date'], inplace=True)

# Group by 'USUBJID' and 'RoiNumber' and filter groups with exactly 2 entries
grouped = oct.groupby(['USUBJID', 'RoiNumber']).filter(lambda x: len(x) == 2)

# Assign 'pre' and 'post' labels based on 'study_date'
grouped['Treatment'] = grouped.groupby(['USUBJID', 'RoiNumber'])['study_date'].transform(lambda x: ['pre', 'post'] if x.iloc[0] < x.iloc[1] else ['post', 'pre'])

# Update the original dataframe to keep only the matched pairs with the new 'Treatment' column
oct_pre = grouped[grouped['Treatment'] == 'pre']
oct_post = grouped[grouped['Treatment'] == 'post']

# %% Generate data (lesion level)

rad_data = oct_pre.copy()
mu = 5
data = []
for i in range(N):
    # Generate random number of lesions with poisson distribution (min 1 and max 30)
    n_lesions = np.random.poisson(mu)
    while n_lesions < 1 or n_lesions > 30:
        n_lesions = np.random.poisson(mu)
    for j in range(n_lesions):
        # take a random index
        ind = random.choice(rad_data.index)
        # Generate diameter (pre-treatment)
        diameter_pre = rad_data['original_shape_Maximum2DDiameterSlice'][ind]
        # Generate diameter change
        diameter_change = trunc_normal_dist.rvs(size=1)[0]
        # Generate diameter (post-treatment)
        diameter_post = diameter_pre + diameter_pre * diameter_change
        # Generate location tag
        location = rad_data['LABEL'][ind]
        # Volume (contoured)
        volume = rad_data['original_shape_VoxelVolume'][ind]
        # Other data
        diameter_3Dmax = rad_data['original_shape_Maximum3DDiameter'][ind]
        diameter_majorAx = rad_data['original_shape_MajorAxisLength'][ind]
        diameter_minorAx = rad_data['original_shape_MinorAxisLength'][ind]

        # Append to data
        data.append([i, j, diameter_pre, diameter_change, diameter_post, location, volume, diameter_3Dmax, diameter_majorAx, diameter_minorAx])

# Convert to DataFrame
synth_lesions_oct = pd.DataFrame(data, columns=['patient', 'lesion', 'diameter-pre', 'diameter_change', 'diameter-post', 'location', 'volume (cc); contoured', 'diameter_3Dmax', 'diameter_majorAx', 'diameter_minorAx'])
synth_lesions_oct['patient'] = synth_lesions_oct['patient'].astype(str)
patients_oct, counts_oct = np.unique(synth_lesions_oct['patient'], return_counts=True)
# %%
from scipy.stats import binned_statistic

plt.rcParams.update({'font.size': 16})

# Concatenate diameter and volume measurements
diameters = np.concatenate([oct_pre['original_shape_Maximum2DDiameterSlice'].values, synth_lesions['diameter-pre'].values]) / 10
volumes = np.concatenate([oct_pre['original_shape_VoxelVolume'].values, synth_lesions['volume (cc); contoured'].values])

expected_volume = 4/3 * np.pi * (np.linspace(0,25,100)/2)**3

# remove any points where the volume is >100 if the diameter is < 2
inds = ~np.logical_and(diameters <= 3, volumes >= 100)
diameters = diameters[inds]
volumes = volumes[inds]

volume_variation = volumes - 4/3 * np.pi * (diameters/2)**3

# Calculate the mean and standard deviation of the volume variation
# Calculate the mean and standard deviation of the volume variation as a function of diameter
bin_means, bin_edges, binnumber = binned_statistic(diameters, volume_variation, statistic='mean', bins=20)
bin_std, _, _ = binned_statistic(diameters, volume_variation, statistic='std', bins=20)

# Interpolate to get a smooth curve
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
mean_volume_variation = np.interp(np.linspace(0, 25, 100), bin_centers, bin_means)
std_volume_variation = np.interp(np.linspace(0, 25, 100), bin_centers, bin_std)

# Calculate the confidence intervals
confidence_interval_upper = expected_volume + 1.96 * std_volume_variation
confidence_interval_lower = expected_volume - 1.96 * std_volume_variation

# Scatter plot of volume versus diameter with confidence intervals
plt.figure(figsize=(8, 4))
plt.scatter(diameters, volumes, alpha=0.5, label='Observed volume')
plt.plot(np.linspace(0, 25, 100), expected_volume, color='red', label='Expected volume')
# plt.fill_between(np.linspace(0, 25, 100), confidence_interval_lower, confidence_interval_upper, color='red', alpha=0.2, label='95% Confidence Interval')
plt.ylim(0, 2000)
plt.xlim(0, 20)
plt.xlabel('Diameter (cm)')
plt.ylabel(r'Volume ($cm^3$)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
sns.despine(trim=True, offset=5)
plt.show()


# %%
# calculate the variation in volume as a function of the diameter
volume_variation = volumes - 4/3 * np.pi * (diameters/2)**3

# Scatter plot of volume versus diameter
plt.figure(figsize=(8, 4))
plt.scatter(diameters, volumes, alpha=0.5, label='Observed volume')
plt.plot(np.linspace(0, 25, 100), expected_volume, color='red', label='Expected volume')
plt.ylim(0, 2000)
plt.xlim(0, 20)
plt.xlabel('Diameter (cm)')
plt.ylabel(r'Volume ($cm^3$)')
# plt.title('Scatter plot of Volume vs Diameter')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(loc='upper left')
sns.despine()
plt.show()

# Residuals plot of volume variation
plt.figure(figsize=(8, 4))
plt.scatter(diameters, volume_variation, alpha=0.5, label='Volume Variation')
plt.axhline(0, color='red', linestyle='--', label='Expected Volume')
plt.ylim(-1000, 1000)
plt.xlim(0, 20)
plt.xlabel('Diameter (cm)')
plt.ylabel('Volume variation ($cm^3$)')
# plt.title('Residuals plot of Volume variation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.show()
# Scatter plot of volume versus diameter
# plt.scatter(diameters, volumes, alpha=0.5, label='Observed volume')
# plt.plot(np.linspace(0,25,100), expected_volume, color='red', label='Expected volume')
# plt.ylim(0, 2000)
# plt.xlim(0, 20)
# plt.xlabel('Diameter (cm)')
# plt.ylabel(r'Volume ($cm^3$)')
# # plt.title('Scatter plot of Volume vs Diameter')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# sns.despine(trim=True,offset=5)
# plt.show()
# %%

