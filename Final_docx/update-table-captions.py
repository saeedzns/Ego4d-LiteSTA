import re

# Table descriptions based on context
table_descriptions = {
    '2.1': 'Baseline methods performance on Ego4D-STA v2',
    '7.1': 'Weighted vs unweighted checkpoint comparison',
    '7.2': 'Run categories and selection criteria', 
    '7.3': 'Backbone and pretraining configurations',
    '7.4': 'ResNet18 vs VideoMAE performance comparison',
    '8.1': 'Track A recall at different K values',
    '8.2': 'Track B baseline performance metrics',
    '8.3': 'Track B vs Track C efficiency comparison',
    '8.4': 'Token pruning configurations and results',
}

# Read file
with open('thesis.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

result = []
for i, line in enumerate(lines):
    # Update table captions with proper descriptions
    if line.startswith('**Table '):
        match = re.match(r'\*\*Table (\d+\.\d+)', line)
        if match:
            table_id = match.group(1)
            desc = table_descriptions.get(table_id, 'Summary')
            result.append(f'**Table {table_id} - {desc}**\n')
            continue
    result.append(line)

with open('thesis.md', 'w', encoding='utf-8') as f:
    f.writelines(result)

print(f'✓ Updated table captions with descriptions')
