import json

# Load the JSON file
with open('all_experiments_results_withprob_20260203_025359.json', 'r') as f:
    data = json.load(f)

# Extract threshold values from the first entry
first_entry = data[0]
thresholds = sorted(first_entry['metrics_per_threshold'].keys(), key=float)

# Group prompts by their text and assign IDs
prompt_to_id = {}
id_to_prompt = {}
current_id = 1

for entry in data:
    # Use the initial_description as the unique identifier for each prompt
    prompt_key = entry['prompt_text']['initial_description']
    
    if prompt_key not in prompt_to_id:
        prompt_to_id[prompt_key] = current_id
        id_to_prompt[current_id] = entry['prompt_text']
        current_id += 1

# Start building the LaTeX document
latex_lines = []
latex_lines.append('\\documentclass{article}')
latex_lines.append('\\usepackage{booktabs}')
latex_lines.append('\\usepackage[margin=0.5in, left=0.3in]{geometry}')
latex_lines.append('')
latex_lines.append('\\begin{document}')
latex_lines.append('')

# Function to generate a metric table
def generate_metric_table(metric_name, metric_key):
    metric_headers = ['Prompt ID', 'Chunk Size (frames)'] + [f'{metric_name}@{t}' for t in thresholds] + ['Average']
    
    table_lines = []
    table_lines.append('\\begin{table}[h]')
    table_lines.append('\\centering')
    table_lines.append('\\tiny')
    table_lines.append('\\begin{tabular}{' + 'c' * len(metric_headers) + '}')
    table_lines.append('\\toprule')
    
    # Add header row
    header_row = '\\textbf{' + '} & \\textbf{'.join(metric_headers) + '} \\\\'
    table_lines.append(header_row)
    table_lines.append('\\midrule')
    
    # Add data rows
    for entry in data:
        prompt_key = entry['prompt_text']['initial_description']
        prompt_id = prompt_to_id[prompt_key]
        chunk_size = entry['chunk_size']
        
        row_data = [str(prompt_id), str(chunk_size)]
        
        # Extract metric scores for each threshold and calculate average
        metric_scores = []
        for threshold in thresholds:
            metric_score = entry['metrics_per_threshold'][threshold][metric_key]
            metric_scores.append(metric_score)
            row_data.append(f'{metric_score:.3f}')
        
        # Calculate and add average
        average_metric = sum(metric_scores) / len(metric_scores)
        row_data.append(f'{average_metric:.3f}')
        
        row_string = ' & '.join(row_data) + ' \\\\'
        table_lines.append(row_string)
    
    # Close the table
    table_lines.append('\\bottomrule')
    table_lines.append('\\end{tabular}')
    table_lines.append(f'\\caption{{{{\\textbf{{{metric_name}}} metrics across different prompt configurations and chunk sizes}}}}')
    table_lines.append(f'\\label{{tab:{metric_key}}}')
    table_lines.append('\\end{table}')
    table_lines.append('')
    
    return table_lines

# Generate F1 table
latex_lines.extend(generate_metric_table('F1', 'f1'))

# Generate Precision table
latex_lines.extend(generate_metric_table('Precision', 'precision'))

# Generate Recall table
latex_lines.extend(generate_metric_table('Recall', 'recall'))

# Add note
latex_lines.append('\\textbf{Note:} Evaluation is performed on a subset of 100 videos.')
latex_lines.append('')# Add prompt mapping section
latex_lines.append('')
latex_lines.append('\\section*{Prompt Descriptions}')
latex_lines.append('')

for prompt_id in sorted(id_to_prompt.keys()):
    prompt_text = id_to_prompt[prompt_id]
    latex_lines.append(f'\\textbf{{Prompt {prompt_id}:}}')
    latex_lines.append('\\begin{itemize}')
    latex_lines.append(f"  \\item \\textit{{Initial description:}} {prompt_text['initial_description']}")
    # Escape braces in validity_check
    validity_check = prompt_text['validity_check'].replace('{', '\\{').replace('}', '\\}').replace('previous_description', 'previous\\_description')
    latex_lines.append(f"  \\item \\textit{{Validity check:}} {validity_check}")
    latex_lines.append(f"  \\item \\textit{{Update description:}} {prompt_text['update_description']}")
    latex_lines.append('\\end{itemize}')
    latex_lines.append('')

# Add document closing
latex_lines.append('\\end{document}')

# Print the LaTeX code
latex_code = '\n'.join(latex_lines)
print(latex_code)

# Also save to a file
with open('metrics_table09.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX table saved to 'metrics_table09.tex'")