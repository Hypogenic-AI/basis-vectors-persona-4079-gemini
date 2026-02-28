import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    with open("results/metrics.json", "r") as f:
        data = json.load(f)
    
    # Layer 14 alignments
    layer = "14"
    alignments = data[layer]["pc1_alignments"]
    
    df = pd.DataFrame(list(alignments.items()), columns=['Behavior', 'Alignment'])
    df = df.sort_values(by='Alignment', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Alignment', y='Behavior', palette='viridis')
    plt.title(f'Alignment of Behaviors with PC1 (Layer {layer})')
    plt.axvline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig('results/pc1_alignments_bar.png')
    plt.close()

if __name__ == "__main__":
    main()
