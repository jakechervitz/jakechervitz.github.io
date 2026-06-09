"""
Bellabeat Marketing Analysis — Google Data Analytics Capstone
=============================================================

A walkthrough of the FitBit Fitness Tracker dataset (Kaggle, public domain)
to find behavioral trends worth turning into Bellabeat app marketing
suggestions.

Author: Jake
Tools:  Python 3, pandas, numpy, matplotlib, seaborn

Note on AI assistance:
    This analysis was completed with help from Claude (Anthropic's AI
    assistant) for code structure, technical guidance, and writing polish.
    All analytical decisions and interpretations are mine; I ran every line
    of code and validated the findings against the data.

How to use:
    Run cells one at a time in PyCharm (Ctrl+Alt+E sends a # %% block to the
    Python Console) or in any editor that supports cell markers. Running the
    whole file top to bottom also works — every step is sequential and
    self-contained.

Folder layout assumed:
    Case Study 2/
    ├── 01_raw_data/
    │   ├── mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/
    │   └── mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/
    ├── 02_cleaned_data/      <- cleaned CSVs land here
    ├── 03_analysis/          <- this script lives here
    ├── 04_visualizations/    <- chart PNGs land here
    └── 05_deliverables/      <- final report and deck
"""

# %% Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path

# Display settings — show full columns when previewing dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Paths. Change BASE_DIR if your project lives somewhere else.
BASE_DIR     = Path(r"C:\Users\jakec\Documents\Coursera Case Studies\Case Study 2")
RAW_DIR      = BASE_DIR / "01_raw_data"
CLEAN_DIR    = BASE_DIR / "02_cleaned_data"
VIZ_DIR      = BASE_DIR / "04_visualizations"

# The dataset comes in two folders — March 12 to April 11, then April 12 to
# May 12, 2016. Each parent folder has a nested folder inside.
FOLDER_1 = RAW_DIR / "mturkfitbit_export_3.12.16-4.11.16" / "Fitabase Data 3.12.16-4.11.16"
FOLDER_2 = RAW_DIR / "mturkfitbit_export_4.12.16-5.12.16" / "Fitabase Data 4.12.16-5.12.16"

# Make output dirs if they don't exist yet
CLEAN_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# Sanity check — both folders should resolve
print(f"Folder 1 exists: {FOLDER_1.exists()}")
print(f"Folder 2 exists: {FOLDER_2.exists()}")


# %% Inventory the files
# Quick recon: what CSVs are in each folder, and what overlaps?
folder_1_files = sorted(FOLDER_1.glob("*.csv"))
folder_2_files = sorted(FOLDER_2.glob("*.csv"))

print(f"\nFolder 1: {len(folder_1_files)} CSVs")
print(f"Folder 2: {len(folder_2_files)} CSVs")

files_1 = {f.name for f in folder_1_files}
files_2 = {f.name for f in folder_2_files}
print(f"\nIn both folders: {len(files_1 & files_2)} files")
print(f"Only in Folder 2: {sorted(files_2 - files_1)}")
# Note: sleepDay_merged.csv is only in Folder 2, which means our sleep
# analysis is limited to those 31 days.


# %% Load the four files we'll actually use
# Daily activity — the workhorse file. Combine both folders.
daily_activity_1 = pd.read_csv(FOLDER_1 / "dailyActivity_merged.csv")
daily_activity_2 = pd.read_csv(FOLDER_2 / "dailyActivity_merged.csv")
daily_activity   = pd.concat([daily_activity_1, daily_activity_2], ignore_index=True)

# Sleep — Folder 2 only
sleep_day = pd.read_csv(FOLDER_2 / "sleepDay_merged.csv")

# Hourly steps — combine both folders
hourly_steps_1 = pd.read_csv(FOLDER_1 / "hourlySteps_merged.csv")
hourly_steps_2 = pd.read_csv(FOLDER_2 / "hourlySteps_merged.csv")
hourly_steps   = pd.concat([hourly_steps_1, hourly_steps_2], ignore_index=True)

# Weight log — combine both folders. (Mostly to show how few people use it.)
weight_log_1 = pd.read_csv(FOLDER_1 / "weightLogInfo_merged.csv")
weight_log_2 = pd.read_csv(FOLDER_2 / "weightLogInfo_merged.csv")
weight_log   = pd.concat([weight_log_1, weight_log_2], ignore_index=True)

# Quick summary of what we have
print("\nLoaded data:")
print(f"  Daily activity: {len(daily_activity):>6} rows, "
      f"{daily_activity['Id'].nunique()} unique users")
print(f"  Sleep day:      {len(sleep_day):>6} rows, "
      f"{sleep_day['Id'].nunique()} unique users")
print(f"  Hourly steps:   {len(hourly_steps):>6} rows, "
      f"{hourly_steps['Id'].nunique()} unique users")
print(f"  Weight log:     {len(weight_log):>6} rows, "
      f"{weight_log['Id'].nunique()} unique users")


# %% Investigate duplicates BEFORE dropping anything
# This step matters — blind drop_duplicates() could erase real data.
print("\nDuplicate rows by (Id, date) before cleaning:")
print(f"  Daily activity: {daily_activity.duplicated(subset=['Id','ActivityDate']).sum()}")
print(f"  Sleep day:      {sleep_day.duplicated(subset=['Id','SleepDay']).sum()}")
print(f"  Hourly steps:   {hourly_steps.duplicated(subset=['Id','ActivityHour']).sum()}")

# Look at what daily_activity duplicates actually are
dup_mask = daily_activity.duplicated(subset=['Id','ActivityDate'], keep=False)
print(f"\nDaily activity duplicates are all on: "
      f"{daily_activity.loc[dup_mask, 'ActivityDate'].unique()}")
# All on 4/12 — turns out Folder 1's extraction got cut off mid-day, so
# Folder 1 has partial-day records and Folder 2 has the full day. We want
# the full-day records.


# %% Clean: drop partial-day records, then true duplicates
# Daily activity: when there's a duplicate (Id, date), keep the row with
# the higher TotalSteps. The partial-day rows always have lower step counts
# than the full-day rows because they cover fewer hours.
before = len(daily_activity)
daily_activity = (
    daily_activity
    .sort_values('TotalSteps', ascending=False)            # higher steps first
    .drop_duplicates(subset=['Id','ActivityDate'], keep='first')
    .sort_values(['Id','ActivityDate'])                    # restore logical order
    .reset_index(drop=True)
)
print(f"\nDaily activity: dropped {before - len(daily_activity)} partial-day rows")

# Sleep: 3 truly identical duplicate rows. Just drop them.
before = len(sleep_day)
sleep_day = sleep_day.drop_duplicates().reset_index(drop=True)
print(f"Sleep day:      dropped {before - len(sleep_day)} duplicate rows")

# Hourly steps: dupes are also from the 4/12 cutoff issue, but in this case
# both folders captured the same hours with the same values. Safe to drop.
before = len(hourly_steps)
hourly_steps = hourly_steps.drop_duplicates().reset_index(drop=True)
print(f"Hourly steps:   dropped {before - len(hourly_steps)} duplicate rows")


# %% Convert date columns to proper datetime
# Specifying the format keeps things fast and avoids parser warnings.
daily_activity['ActivityDate'] = pd.to_datetime(
    daily_activity['ActivityDate'], format='%m/%d/%Y'
)
sleep_day['SleepDay'] = pd.to_datetime(
    sleep_day['SleepDay'], format='%m/%d/%Y %I:%M:%S %p'
).dt.date
hourly_steps['ActivityHour'] = pd.to_datetime(
    hourly_steps['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p'
)

print("\nDate ranges:")
print(f"  Daily activity: {daily_activity['ActivityDate'].min().date()} "
      f"to {daily_activity['ActivityDate'].max().date()}")
print(f"  Sleep day:      {sleep_day['SleepDay'].min()} "
      f"to {sleep_day['SleepDay'].max()}")
print(f"  Hourly steps:   {hourly_steps['ActivityHour'].min()} "
      f"to {hourly_steps['ActivityHour'].max()}")


# %% Filter out non-wear days from activity data
# A day with under 100 steps is almost certainly a day the user didn't wear
# the device. Keeping these in averages would drag everything down.
# We keep daily_activity intact (for engagement metrics) and create a
# filtered version for activity analysis.
print("\nNon-wear day analysis:")
print(f"  Days with 0 steps:        {(daily_activity['TotalSteps'] == 0).sum()}")
print(f"  Days with < 100 steps:    {(daily_activity['TotalSteps'] < 100).sum()}")
print(f"  Total days in dataset:    {len(daily_activity)}")

daily_activity_worn = daily_activity[daily_activity['TotalSteps'] >= 100].copy()
print(f"  After non-wear filter:    {len(daily_activity_worn)}")
# 11% of recorded days were non-wear — itself a finding about engagement.


# %% Classify users into activity tiers
# Using Tudor-Locke step thresholds from public-health research instead of
# arbitrary cutoffs.
user_avg_steps = (
    daily_activity_worn
    .groupby('Id')['TotalSteps']
    .mean()
    .sort_values()
)

def classify_user(avg_steps):
    if avg_steps < 5000:
        return "Sedentary"
    elif avg_steps < 7500:
        return "Lightly Active"
    elif avg_steps < 10000:
        return "Fairly Active"
    else:
        return "Very Active"

user_segments = user_avg_steps.apply(classify_user)
print("\nUser activity tiers:")
print(user_segments.value_counts())
print("\nAs percentages:")
print((user_segments.value_counts(normalize=True) * 100).round(1))


# %% Daily activity composition (the 79% sedentary finding)
intensity_cols = [
    'VeryActiveMinutes', 'FairlyActiveMinutes',
    'LightlyActiveMinutes', 'SedentaryMinutes'
]
avg_minutes = daily_activity_worn[intensity_cols].mean()
total_minutes = avg_minutes.sum()

print("\nAverage minutes per day on worn days:")
print(avg_minutes.round(1))
print(f"\nTotal tracked minutes/day: {total_minutes:.0f} ({total_minutes/60:.1f} hrs)")
print("\nPercentage breakdown:")
print((avg_minutes / total_minutes * 100).round(1))


# %% Sleep analysis
# Convert to hours for readability + add a sleep efficiency column.
sleep_day['HoursAsleep']     = sleep_day['TotalMinutesAsleep'] / 60
sleep_day['HoursInBed']      = sleep_day['TotalTimeInBed'] / 60
sleep_day['SleepEfficiency'] = (sleep_day['TotalMinutesAsleep']
                                / sleep_day['TotalTimeInBed'] * 100).round(1)

print("\nSleep summary:")
print(sleep_day[['HoursAsleep','HoursInBed','SleepEfficiency']].describe().round(2))

user_avg_sleep = sleep_day.groupby('Id')['HoursAsleep'].mean()
print(f"\nUsers averaging >= 7 hours: "
      f"{(user_avg_sleep >= 7).sum()} of {len(user_avg_sleep)}")
print(f"Users averaging < 6 hours:  "
      f"{(user_avg_sleep < 6).sum()} of {len(user_avg_sleep)}")
# Average is right at 7 hrs, which is misleading. Only 46% actually meet
# 7+; a third are getting under 6. Sleep efficiency is high though, so the
# issue is bedtime allocation, not sleep quality.


# %% Activity vs sleep — does sitting more correlate with sleeping less?
# Need consistent date types on both sides to merge.
daily_activity_worn['ActivityDate'] = (
    pd.to_datetime(daily_activity_worn['ActivityDate']).dt.date
)

merged = daily_activity_worn.merge(
    sleep_day,
    left_on=['Id','ActivityDate'],
    right_on=['Id','SleepDay'],
    how='inner'
)

print(f"\nMerged dataset: {len(merged)} user-days, "
      f"{merged['Id'].nunique()} users")

correlations = merged[[
    'TotalSteps','VeryActiveMinutes','SedentaryMinutes','TotalMinutesAsleep'
]].corr()['TotalMinutesAsleep'].round(3)
print("\nCorrelation with TotalMinutesAsleep:")
print(correlations)
# Sedentary minutes correlates -0.64 with sleep — strongest signal in the
# dataset. Steps barely correlate (-0.20). It's specifically sitting time
# that lines up with bad sleep, not lack of fitness.


# %% Hourly step pattern — when are users most active?
hourly_steps['Hour'] = hourly_steps['ActivityHour'].dt.hour
avg_steps_by_hour = hourly_steps.groupby('Hour')['StepTotal'].mean().round(0)

print("\nAverage steps by hour:")
print(avg_steps_by_hour)
print(f"\nPeak hour: {avg_steps_by_hour.idxmax()}:00 "
      f"({int(avg_steps_by_hour.max())} steps)")
print(f"Lowest:    {avg_steps_by_hour.idxmin()}:00 "
      f"({int(avg_steps_by_hour.min())} steps)")
# Peak at 7 PM, clear afternoon lull from 1-4 PM, post-8 PM drop-off.


# %% Engagement / wear consistency
days_per_user = daily_activity.groupby('Id').size().sort_values()
print(f"\nDays of data per user (out of 61 possible):")
print(days_per_user.describe().round(1))
print(f"\nUsers with fewer than 30 days: {(days_per_user < 30).sum()}")
# Average user only has 39 of 61 days — engagement decays fast.


# %% Export cleaned data for downstream use
daily_activity_worn.to_csv(CLEAN_DIR / "daily_activity_cleaned.csv", index=False)
sleep_day.to_csv(CLEAN_DIR / "sleep_day_cleaned.csv", index=False)
hourly_steps.to_csv(CLEAN_DIR / "hourly_steps_cleaned.csv", index=False)
merged.to_csv(CLEAN_DIR / "activity_sleep_merged.csv", index=False)
print(f"\nCleaned data exported to: {CLEAN_DIR}")


# %% Plotting setup
# Bellabeat-ish palette: warm coral primary, sage secondary, deep slate dark
BELLA = {
    'primary':   '#E07A5F',  # warm coral
    'secondary': '#81B29A',  # sage green
    'accent':    '#F2CC8F',  # warm yellow
    'dark':      '#3D405B',  # deep slate
    'light':     '#F4F1DE'   # cream
}

sns.set_style('whitegrid')
plt.rcParams['figure.figsize']     = (10, 6)
plt.rcParams['font.size']          = 11
plt.rcParams['axes.titlesize']     = 14
plt.rcParams['axes.titleweight']   = 'bold'
plt.rcParams['axes.spines.top']    = False
plt.rcParams['axes.spines.right']  = False


# %% Chart 1 — User segments
segment_order  = ['Sedentary', 'Lightly Active', 'Fairly Active', 'Very Active']
segment_counts = user_segments.value_counts().reindex(segment_order)

fig, ax = plt.subplots()
bars = ax.bar(segment_counts.index, segment_counts.values,
              color=[BELLA['primary'], BELLA['accent'],
                     BELLA['secondary'], BELLA['dark']])

for bar, count in zip(bars, segment_counts.values):
    pct = count / segment_counts.sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10)

ax.set_title('Users are evenly split across activity levels — no single "typical" user',
             loc='left')
ax.set_ylabel('Number of users')
ax.set_ylim(0, max(segment_counts.values) + 2)
plt.tight_layout()
plt.savefig(VIZ_DIR / "01_user_segments.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Chart 2 — How users spend their tracked day (the 79% sedentary chart)
labels  = ['Sedentary', 'Lightly Active', 'Fairly Active', 'Very Active']
minutes = avg_minutes.values
colors  = [BELLA['primary'], BELLA['accent'], BELLA['secondary'], BELLA['dark']]

fig, ax = plt.subplots(figsize=(10, 5))
left = 0
for i, (lbl, mins, c) in enumerate(zip(labels, minutes, colors)):
    pct = mins / minutes.sum() * 100
    ax.barh(['Average Day'], [mins], left=[left], color=c,
            label=f'{lbl} ({pct:.0f}%)')
    left += mins

ax.set_title('79% of the average tracked day is sedentary', loc='left')
ax.set_xlabel('Minutes per day')
ax.set_xlim(0, 1300)
ax.legend(loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig(VIZ_DIR / "02_daily_composition.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Chart 3 — Hourly step pattern
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(avg_steps_by_hour.index, avg_steps_by_hour.values,
        color=BELLA['primary'], linewidth=2.5, marker='o', markersize=5)
ax.fill_between(avg_steps_by_hour.index, avg_steps_by_hour.values,
                color=BELLA['primary'], alpha=0.15)

# Annotate the peak
peak_hour  = avg_steps_by_hour.idxmax()
peak_value = avg_steps_by_hour.max()
ax.annotate(f'Peak: 7 PM\n{int(peak_value)} steps',
            xy=(peak_hour, peak_value),
            xytext=(peak_hour - 5, peak_value + 80),
            fontsize=10, color=BELLA['dark'],
            arrowprops=dict(arrowstyle='->', color=BELLA['dark']))

# Highlight the afternoon lull
ax.axvspan(13, 16, alpha=0.1, color=BELLA['secondary'])
ax.text(14.5, 100, 'Afternoon lull\n(notification window)',
        ha='center', fontsize=9, color=BELLA['dark'], style='italic')

ax.set_title('Activity peaks at 7 PM, with an afternoon lull from 1–4 PM', loc='left')
ax.set_xlabel('Hour of day')
ax.set_ylabel('Average steps')
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)])
plt.tight_layout()
plt.savefig(VIZ_DIR / "03_hourly_pattern.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Chart 4 — Sleep distribution per user
user_avg_sleep_sorted = sleep_day.groupby('Id')['HoursAsleep'].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [
    BELLA['primary']   if hrs < 6 else
    BELLA['accent']    if hrs < 7 else
    BELLA['secondary']
    for hrs in user_avg_sleep_sorted.values
]
ax.barh(range(len(user_avg_sleep_sorted)), user_avg_sleep_sorted.values, color=bar_colors)

ax.axvline(7, color=BELLA['dark'], linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(7.05, len(user_avg_sleep_sorted) - 1, '7-hour minimum',
        fontsize=9, color=BELLA['dark'], style='italic')

ax.set_title('1 in 3 sleep-tracking users averages less than 6 hours per night',
             loc='left')
ax.set_xlabel('Average hours of sleep per night')
ax.set_ylabel('Users (anonymized, sorted)')
ax.set_yticks([])
ax.set_xlim(0, 10)

legend_elements = [
    Patch(facecolor=BELLA['primary'],   label='< 6 hrs (sleep-deprived)'),
    Patch(facecolor=BELLA['accent'],    label='6–7 hrs (under recommendation)'),
    Patch(facecolor=BELLA['secondary'], label='≥ 7 hrs (meets recommendation)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig(VIZ_DIR / "04_sleep_distribution.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Chart 5 — Sedentary vs sleep (the strongest signal)
sed_hours   = merged['SedentaryMinutes'] / 60
sleep_hours = merged['TotalMinutesAsleep'] / 60

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(sed_hours, sleep_hours, color=BELLA['primary'],
           alpha=0.5, s=40, edgecolor='white', linewidth=0.5)

# Regression line
z = np.polyfit(sed_hours, sleep_hours, 1)
p = np.poly1d(z)
x_line = np.linspace(sed_hours.min(), sed_hours.max(), 100)
ax.plot(x_line, p(x_line), color=BELLA['dark'], linewidth=2, linestyle='--')

ax.text(0.05, 0.05, 'Correlation: −0.64\n(strong negative)',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor=BELLA['light'], edgecolor='none'))

ax.set_title('Users who sit more sleep less — the strongest relationship in the data',
             loc='left')
ax.set_xlabel('Sedentary hours per day')
ax.set_ylabel('Hours of sleep')
plt.tight_layout()
plt.savefig(VIZ_DIR / "05_sedentary_vs_sleep.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Chart 6 — Wear consistency
TOTAL_POSSIBLE_DAYS = 61

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [
    BELLA['primary']   if d < 30 else
    BELLA['accent']    if d < 50 else
    BELLA['secondary']
    for d in days_per_user.values
]
ax.barh(range(len(days_per_user)), days_per_user.values, color=bar_colors)

ax.axvline(TOTAL_POSSIBLE_DAYS, color=BELLA['dark'],
           linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(TOTAL_POSSIBLE_DAYS + 0.5, len(days_per_user) - 1,
        f'{TOTAL_POSSIBLE_DAYS} days possible',
        fontsize=9, color=BELLA['dark'], style='italic')

ax.set_title('Users only had data for 39 of 61 days on average — engagement decays quickly',
             loc='left')
ax.set_xlabel('Days with recorded data')
ax.set_ylabel('Users (anonymized, sorted)')
ax.set_yticks([])
ax.set_xlim(0, 70)

legend_elements = [
    Patch(facecolor=BELLA['primary'],   label='< 30 days (low engagement)'),
    Patch(facecolor=BELLA['accent'],    label='30–49 days (moderate)'),
    Patch(facecolor=BELLA['secondary'], label='50+ days (high engagement)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig(VIZ_DIR / "06_wear_consistency.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nAll charts saved to: {VIZ_DIR}")
print("\nAnalysis complete.")
