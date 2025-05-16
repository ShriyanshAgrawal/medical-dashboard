import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
csv_path = "test.csv"  # Replace with full path if needed
df = pd.read_csv(csv_path)

# Ask user to input the .parquet file path
parquet_path = input("Upload and enter the path to the .parquet file (e.g., /mnt/data/abc123.parquet): ").strip()

# Extract just the file name without extension
filename = os.path.splitext(os.path.basename(parquet_path))[0]

# Check if filename matches any spectrogram_id
if filename in df['spectrogram_id'].astype(str).values:
    # Get the row that matches the filename
    row = df[df['spectrogram_id'].astype(str) == filename].iloc[0]

    # Extract vote columns
    votes = {
        'seizure_vote': row['seizure_vote'],
        'lpd_vote': row['lpd_vote'],
        'gpd_vote': row['gpd_vote'],
        'lrda_vote': row['lrda_vote'],
        'grda_vote': row['grda_vote'],
        'other_vote': row['other_vote']
    }
    expert_result = row['expert_consensus']

    # Plot a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(votes.keys(), votes.values(), color='steelblue')
    plt.title(f"Spectograph\nExpert Consensus: {expert_result}", fontsize=12)
    expert_consensus = row['expert_consensus']
    plt.text(0.5, max(votes.values()) * 1.1, f"Expert Consensus: {expert_consensus}",
             ha='center', va='bottom', fontsize=12, color='darkred', weight='bold', transform=plt.gca().transAxes)

    plt.xticks()
    plt.tight_layout()
    plt.show()

    # Print expert consensus
    print("Expert Consensus:", row['expert_consensus'])

else:
    print(f"No matching spectrogram_id found for file: {filename}")
