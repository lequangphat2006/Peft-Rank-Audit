""""""Update README trạng thái pilot.""""""
import pandas as pd
import re, os
from datetime import date

MERGED = 'results/pilot/results_C1_merged.csv'
N_TOTAL = 125

if not os.path.exists(MERGED):
    print(f'Không thấy {MERGED}')
    exit(1)

df = pd.read_csv(MERGED)
n_done = len(df)

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

status = f'✅ {n_done}/{N_TOTAL}' if n_done >= N_TOTAL else f'⏳ {n_done}/{N_TOTAL}'
content = re.sub(
    r'\| Pilot C1 \(125 lượt\) \| [^|]+ \|[^|]+\|',
    f'| Pilot C1 (125 lượt) | {status} | {date.today()} |',
    content
)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'README updated: {n_done}/{N_TOTAL}')
