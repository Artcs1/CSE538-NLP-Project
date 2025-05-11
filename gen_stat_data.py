import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import numpy as np
import jieba
from transformers import AutoTokenizer

# Ensure the stats directory exists
os.makedirs('stats', exist_ok=True)

# Load annotations_translated.json
with open('annotations_translated.json', 'r', encoding='utf-8') as file:

    data = json.load(file)

# Filtering records with 'translated' field and ignoring non-string texts
data = [item for item in data if 'translated' in item and isinstance(item.get('text'), str)]

# Total records
total_records = len(data)

# Language distribution
languages = [item['language'] for item in data]
lang_distribution = Counter(languages)

# Label-wise statistics per language
label_stats = {'Language': [], 'Label 1': [], 'Label 0': [], 'Total': []}
for lang in set(languages):
    lang_data = [item for item in data if item['language'] == lang]
    label_1 = sum(1 for item in lang_data if item['label'] == 1)
    label_0 = sum(1 for item in lang_data if item['label'] == 0)
    label_stats['Language'].append(lang)
    label_stats['Label 1'].append(label_1)
    label_stats['Label 0'].append(label_0)
    label_stats['Total'].append(label_1 + label_0)

# 0. Build and sort the base DataFrame of languages only
label_stats_df = pd.DataFrame(label_stats)
label_stats_df = label_stats_df.sort_values(by='Total', ascending=False).reset_index(drop=True)

# 1. Identify numeric columns and compute summaries on language rows only
numeric_cols = ['Label 1', 'Label 0', 'Total']
base        = label_stats_df[numeric_cols]

summary_df = pd.DataFrame([
    {'Language': 'Total', **base.sum().astype(int).to_dict()},
    {'Language': 'Min',   **base.min().astype(int).to_dict()},
    {'Language': 'Avg',   **base.mean().round(1).to_dict()},
    {'Language': 'Max',   **base.max().astype(int).to_dict()},
])

# 2. Append the summary rows below the language rows
label_stats_df = pd.concat([label_stats_df, summary_df], ignore_index=True)

# 3. Allow mixing ints and floats in the same columns
label_stats_df[numeric_cols] = label_stats_df[numeric_cols].astype(object)

# 4. Cast every row except 'Avg' to int, leave 'Avg' as one-decimal floats
mask_avg     = label_stats_df['Language'] == 'Avg'
mask_not_avg = ~mask_avg

label_stats_df.loc[mask_not_avg, numeric_cols] = (
    label_stats_df.loc[mask_not_avg, numeric_cols]
    .astype(int)
)
label_stats_df.loc[mask_avg, numeric_cols] = (
    label_stats_df.loc[mask_avg, numeric_cols]
    .round(1)
    .astype(float)
)

# 5. Finally, write out to CSV
label_stats_df.to_csv('stats/label_stats.csv', index=False)
# ============================================================================



# Character-wise statistics with total columns
char_stats = []
for lang in set(languages):
    texts = [item['text'] for item in data if item['language'] == lang]
    translated_texts = [item['translated'] for item in data if item['language'] == lang]

    char_stats.append([
        lang,
        round(np.mean([len(text) for text in texts]), 1),
        min(len(text) for text in texts),
        max(len(text) for text in texts),
        sum(len(text) for text in texts),
        round(np.mean([len(text) for text in translated_texts]), 1),
        min(len(text) for text in translated_texts),
        max(len(text) for text in translated_texts),
        sum(len(text) for text in translated_texts)
    ])

# Character statistics dataframe with total columns
char_stats_df = pd.DataFrame(char_stats, columns=[
    'Language', 'Original Text Avg', 'Original Text Min', 'Original Text Max', 'Original Text Total',
    'Translated Text Avg', 'Translated Text Min', 'Translated Text Max', 'Translated Text Total'
])
char_stats_df = char_stats_df.sort_values(by='Original Text Total', ascending=False)
char_stats_df.loc['Total'] = [
    'Total',
    round(char_stats_df['Original Text Avg'].mean(), 1),
    int(char_stats_df['Original Text Min'].min()),
    int(char_stats_df['Original Text Max'].max()),
    int(char_stats_df['Original Text Total'].sum()),
    round(char_stats_df['Translated Text Avg'].mean(), 1),
    int(char_stats_df['Translated Text Min'].min()),
    int(char_stats_df['Translated Text Max'].max()),
    int(char_stats_df['Translated Text Total'].sum())
]
char_stats_df.to_csv('stats/char_stats.csv', index=False)
# ============================================================================


# Handling outliers in Character-wise Box Plot
char_lengths = {lang: [len(item['translated']) for item in data if item['language'] == lang] for lang in set(languages)}
plt.figure(figsize=(14, 7))
plt.boxplot(char_lengths.values(), tick_labels=char_lengths.keys(), patch_artist=True, showfliers=False)
plt.title('Character-wise Statistics by Language (Box Plot - Outliers Excluded)')
plt.xlabel('Languages')
plt.ylabel('Character Count')
plt.tight_layout()
plt.savefig('stats/characterwise_boxplot_translate.png')
# ============================================================================

# Language Distribution Pie Chart with Larger Labels, Adjusted % and Distinct Colors
plt.figure(figsize=(10, 10))
colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(lang_distribution)))
wedges, texts, autotexts = plt.pie(
    lang_distribution.values(),
    labels=lang_distribution.keys(),
    autopct='%1.1f%%',
    textprops={'fontsize': 16, 'weight': 'bold'},
    colors=colors,
    startangle=90
)

# Adjusting % text positions slightly towards the edge
for autotext in autotexts:
    autotext.set_position((autotext.get_position()[0] * 1.1, autotext.get_position()[1] * 1.1))

plt.title('Language Distribution Percentage', fontsize=18)
plt.tight_layout()
plt.savefig('stats/language_distribution.png')
# ============================================================================

# 1. Filter out the Total/Min/Avg/Max rows
to_drop = ['Total','Min','Avg','Max']
plot_df = label_stats_df[~label_stats_df['Language'].isin(to_drop)]

# 2. Set Language as index and plot only the real languages
plot_df.set_index('Language')[['Label 1','Label 0']].plot(
    kind='bar',
    stacked=True,
    figsize=(12,6)
)

plt.title('Label Distribution by Language (Stacked)')
plt.xlabel('Languages')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('stats/label_distribution_stacked.png')
plt.close()
# ============================================================================

# Word-wise statistics with min, max, avg, total per language (and Total at end)
word_stats = []
for lang in set(languages):
    texts = [item['text'] for item in data if item['language'] == lang]
    translated_texts = [item['translated'] for item in data if item['language'] == lang]
    
    original_counts   = [len(t.split()) for t in texts]
    translated_counts = [len(t.split()) for t in translated_texts]
    
    word_stats.append([
        lang,
        # total words instead of record count
        sum(original_counts),
        min(original_counts),
        max(original_counts),
        round(np.mean(original_counts), 1),
        sum(translated_counts),
        min(translated_counts),
        max(translated_counts),
        round(np.mean(translated_counts), 1)
    ])

# build dataframe with new column names
word_stats_df = pd.DataFrame(word_stats, columns=[
    'Language',
    'Original Words Total', 'Original Words Min', 'Original Words Max', 'Original Words Avg',
    'Translated Words Total', 'Translated Words Min', 'Translated Words Max', 'Translated Words Avg'
])

# compute the overall Total row
total_texts            = [item['text'] for item in data]
total_translated_texts = [item['translated'] for item in data]
total_orig_counts      = [len(t.split()) for t in total_texts]
total_trans_counts     = [len(t.split()) for t in total_translated_texts]

total_row = {
    'Language':              'Total',
    'Original Words Total':  sum(total_orig_counts),
    'Original Words Min':    min(total_orig_counts),
    'Original Words Max':    max(total_orig_counts),
    'Original Words Avg':    round(np.mean(total_orig_counts), 1),
    'Translated Words Total': sum(total_trans_counts),
    'Translated Words Min':   min(total_trans_counts),
    'Translated Words Max':   max(total_trans_counts),
    'Translated Words Avg':   round(np.mean(total_trans_counts), 1),
}

# append Total at the end
word_stats_df = pd.concat([
    word_stats_df,
    pd.DataFrame([total_row])
], ignore_index=True)

# sort all languages by descending total original words, but keep Total last
top_langs = word_stats_df.iloc[:-1].sort_values(
    by='Original Words Total', ascending=False
)
word_stats_df = pd.concat([top_langs, word_stats_df.iloc[[-1]]], ignore_index=True)

# write out
word_stats_df.to_csv('stats/word_stats.csv', index=False)
# ============================================================================


# Language-wise Record Count Statistics
lang_record_counts = pd.DataFrame.from_dict(lang_distribution, orient='index', columns=['Record Count'])
lang_record_counts = lang_record_counts.sort_values(by='Record Count', ascending=False)
lang_record_counts.loc['Total'] = lang_record_counts.sum()

# Saving Record Count CSV
lang_record_counts.to_csv('stats/language_record_counts.csv', index_label='Language')

# Plotting Language-wise Record Counts
plt.figure(figsize=(12, 6))
lang_record_counts.drop('Total').plot(kind='bar', legend=False, color='skyblue')
plt.title('Language-wise Record Counts')
plt.xlabel('Languages')
plt.ylabel('Record Count')
plt.tight_layout()
plt.savefig('stats/language_record_counts.png')
# ============================================================================


# Generating Word-wise Box Plot (Outliers Excluded)
word_lengths = {lang: [len(item['translated'].split()) for item in data if item['language'] == lang] for lang in set(languages)}
plt.figure(figsize=(14, 7))
plt.boxplot(word_lengths.values(), tick_labels=word_lengths.keys(), patch_artist=True, showfliers=False)
plt.title('Word-wise Statistics by Language (Box Plot - Outliers Excluded)')
plt.xlabel('Languages')
plt.ylabel('Word Count')
plt.tight_layout()
plt.savefig('stats/wordwise_boxplot_translate.png')
# ============================================================================

# 1. Assemble your per-language character counts
langs = sorted(set(languages))
orig_counts = [ 
    [len(item['text']) for item in data if item['language'] == lang] 
    for lang in langs 
]
trans_counts = [ 
    [len(item['translated']) for item in data if item['language'] == lang] 
    for lang in langs 
]

# 2. Compute x positions
x = np.arange(len(langs))
width = 0.35

# 3. Draw the two boxplots *without* outliers, with colors
fig, ax = plt.subplots(figsize=(12,6))

bp1 = ax.boxplot(
    orig_counts,
    positions=x - width/2,
    widths=width,
    showfliers=False,
    patch_artist=True            # <— allow box facecolors
)
bp2 = ax.boxplot(
    trans_counts,
    positions=x + width/2,
    widths=width,
    showfliers=False,
    patch_artist=True
)

# 4. Set distinct colors
orig_color = '#1f77b4'    # blue
trans_color = '#ff7f0e'   # orange

for box in bp1['boxes']:
    box.set(facecolor=orig_color, edgecolor='black')
for whisker in bp1['whiskers']:
    whisker.set(color=orig_color)
for cap in bp1['caps']:
    cap.set(color=orig_color)
for median in bp1['medians']:
    median.set(color='white', linewidth=2)

for box in bp2['boxes']:
    box.set(facecolor=trans_color, edgecolor='black')
for whisker in bp2['whiskers']:
    whisker.set(color=trans_color)
for cap in bp2['caps']:
    cap.set(color=trans_color)
for median in bp2['medians']:
    median.set(color='white', linewidth=2)

# 5. Tweak labels & legend
ax.set_xticks(x)
ax.set_xticklabels(langs, rotation=45, ha='right')
ax.set_ylabel("Character Count")
ax.set_title("Original vs Translated Text Lengths by Language\n(Boxplots without outliers)")
ax.legend(
    [bp1["boxes"][0], bp2["boxes"][0]],
    ["Original", "Translated"],
    loc="upper left"
)

plt.tight_layout()
plt.savefig("stats/char_counts_boxplot_comparison_colored.png", dpi=300, bbox_inches="tight")
plt.close(fig)
# ============================================================================


langs = sorted(set(languages))

# 1. Define a small helper that picks tokenizer by language
def word_count(text, lang, original=False):
    if lang == 'china' and original:
        # Chinese segmentation
        return len(list(jieba.cut(text)))
    else:
        # whitespace splitting for all others
        return len(text.split())

# 2. Build per‐language original & translated word‐count lists
orig_word_counts = [
    [word_count(item['text'], lang, original=True) for item in data if item['language'] == lang]
    for lang in langs
]
trans_word_counts = [
    [word_count(item['translated'], lang) for item in data if item['language'] == lang]
    for lang in langs
]

# 3. Plot side‐by‐side boxplots (outliers excluded, colored)
x = np.arange(len(langs))
width = 0.35
fig, ax = plt.subplots(figsize=(14,7))

bp_orig = ax.boxplot(
    orig_word_counts, positions=x-width/2, widths=width,
    patch_artist=True, showfliers=False
)
bp_trans = ax.boxplot(
    trans_word_counts, positions=x+width/2, widths=width,
    patch_artist=True, showfliers=False
)

# 4. Color them
orig_color, trans_color = '#1f77b4','#ff7f0e'
for b in bp_orig['boxes']:   b.set(facecolor=orig_color, edgecolor='black')
for b in bp_trans['boxes']:  b.set(facecolor=trans_color, edgecolor='black')
for part in ('whiskers','caps'):
    for line in bp_orig[part]:  line.set(color=orig_color)
    for line in bp_trans[part]: line.set(color=trans_color)
for m in bp_orig['medians']:   m.set(color='white', linewidth=2)
for m in bp_trans['medians']:  m.set(color='white', linewidth=2)

# 5. Labels & legend
ax.set_xticks(x)
ax.set_xticklabels(langs, rotation=45, ha='right')
ax.set_ylabel("Word Count")
ax.set_title("Word-wise Statistics by Language (Box Plot — Outliers Excluded)")
ax.legend(
    [bp_orig["boxes"][0], bp_trans["boxes"][0]],
    ["Original", "Translated"],
    loc="upper left"
)

plt.tight_layout()
plt.savefig("stats/wordwise_boxplot_comparison_mixed_tokenizers.png", dpi=300, bbox_inches="tight")
plt.close(fig)
# ============================================================================


# 1. Initialize your HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# 2. Prepare per‐language token‐count lists
langs = sorted(set(languages))
orig_token_counts = [
    [len(tokenizer(item['text'], add_special_tokens=False)['input_ids'])
     for item in data if item['language']==lang]
    for lang in langs
]
trans_token_counts = [
    [len(tokenizer(item['translated'], add_special_tokens=False)['input_ids'])
     for item in data if item['language']==lang]
    for lang in langs
]

# 3. Plot side‐by‐side boxplots (outliers excluded + colored)
x = np.arange(len(langs))
width = 0.35
fig, ax = plt.subplots(figsize=(14,7))

bp_orig = ax.boxplot(
    orig_token_counts,
    positions = x - width/2,
    widths    = width,
    patch_artist=True,
    showfliers=False
)
bp_trans = ax.boxplot(
    trans_token_counts,
    positions = x + width/2,
    widths    = width,
    patch_artist=True,
    showfliers=False
)

# 4. Color styling
orig_color, trans_color = '#1f77b4','#ff7f0e'
for box in bp_orig['boxes']:   box.set(facecolor=orig_color, edgecolor='black')
for whisker in bp_orig['whiskers']: whisker.set(color=orig_color)
for cap in bp_orig['caps']:     cap.set(color=orig_color)
for median in bp_orig['medians']: median.set(color='white', linewidth=2)

for box in bp_trans['boxes']:  box.set(facecolor=trans_color, edgecolor='black')
for whisker in bp_trans['whiskers']: whisker.set(color=trans_color)
for cap in bp_trans['caps']:    cap.set(color=trans_color)
for median in bp_trans['medians']: median.set(color='white', linewidth=2)

# 5. Labels, legend, save
ax.set_xticks(x)
ax.set_xticklabels(langs, rotation=45, ha='right')
ax.set_ylabel("Token Count")
ax.set_title("Token-wise Statistics by Language (Box Plot — Outliers Excluded)")
ax.legend(
    [bp_orig["boxes"][0], bp_trans["boxes"][0]],
    ["Original", "Translated"],
    loc="upper left"
)

plt.tight_layout()
plt.savefig("stats/tokenwise_boxplot_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
# ============================================================================

# 1. Gather per‐language token counts
langs = sorted(set(languages))
stats = []
for lang in langs:
    orig_texts  = [item['text']       for item in data if item['language']==lang]
    trans_texts = [item['translated'] for item in data if item['language']==lang]

    # count tokens (no special tokens) for each string
    orig_tokens  = [len(tokenizer(t, add_special_tokens=False)['input_ids']) for t in orig_texts]
    trans_tokens = [len(tokenizer(t, add_special_tokens=False)['input_ids']) for t in trans_texts]

    stats.append({
        'Language':                 lang,
        'Original Tokens Total':    sum(orig_tokens),
        'Original Tokens Min':      min(orig_tokens),
        'Original Tokens Max':      max(orig_tokens),
        'Original Tokens Avg':      round(np.mean(orig_tokens), 1),
        'Translated Tokens Total':  sum(trans_tokens),
        'Translated Tokens Min':    min(trans_tokens),
        'Translated Tokens Max':    max(trans_tokens),
        'Translated Tokens Avg':    round(np.mean(trans_tokens), 1),
    })

# 2. Build DataFrame
df_token_stats = pd.DataFrame(stats)

# 3. Compute the overall Total row
all_orig_tokens  = [len(tokenizer(item['text'], add_special_tokens=False)['input_ids'])       for item in data]
all_trans_tokens = [len(tokenizer(item['translated'], add_special_tokens=False)['input_ids']) for item in data]
total_row = {
    'Language':                 'Total',
    'Original Tokens Total':    sum(all_orig_tokens),
    'Original Tokens Min':      min(all_orig_tokens),
    'Original Tokens Max':      max(all_orig_tokens),
    'Original Tokens Avg':      round(np.mean(all_orig_tokens), 1),
    'Translated Tokens Total':  sum(all_trans_tokens),
    'Translated Tokens Min':    min(all_trans_tokens),
    'Translated Tokens Max':    max(all_trans_tokens),
    'Translated Tokens Avg':    round(np.mean(all_trans_tokens), 1),
}

# 4. Append Total and save CSV
df_token_stats = pd.concat([df_token_stats, pd.DataFrame([total_row])], ignore_index=True)
df_token_stats.to_csv('stats/token_stats.csv', index=False)

print("✅ All statistics generated and saved under 'stats/' directory.")
