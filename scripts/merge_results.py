""""""Merge nhiều session results thành một file.""""""
import pandas as pd
import glob
import sys

files = sorted(glob.glob('results/pilot/results_C1_session*.csv'))
if not files:
    print('Không tìm thấy file session')
    sys.exit(1)

print(f'Merging {len(files)} files:')
for f in files:
    print(f'  {f}')

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=['seed','config','layer','rank'], keep='last')
df = df.sort_values(['seed','config','layer','rank'])

out = 'results/pilot/results_C1_merged.csv'
df.to_csv(out, index=False)
print(f'Merged: {len(df)} unique rows -> {out}')
print(f'Seeds: {sorted(df["seed"].unique())}')
