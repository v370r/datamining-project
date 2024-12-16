import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

tickers = [
    'NVDA', 
    'AAPL', 
    'UBER', 
    'ADP', 
    'META', 
    'AMD',
    'GOOG',
    'AMC' # For some reason, there is no data for AMC
]

base_df = pd.read_csv('hidden-stock-output/accuracy_data.csv')

for ticker in tickers:
    accuracy_df = base_df[base_df['testing_ticker'] == ticker]
    print(accuracy_df)
    # Extract data from the DataFrame
    x = accuracy_df['num_clusters']
    y = accuracy_df['max_depth']
    z = np.zeros_like(x)  # All bars start from the z=0 plane
    dz = accuracy_df['accuracy']   # The height of the bars corresponds to accuracy

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize accuracy values to map them to colors
    norm = plt.Normalize(vmin=accuracy_df['accuracy'].min(), vmax=accuracy_df['accuracy'].max())
    colors = plt.cm.viridis(norm(accuracy_df['accuracy']))
    # Create a 3D bar plot for each individual data point
    ax.bar3d(x, y, z, 0.5, 0.5, dz, color=colors, zsort='average')

    # Set axis labels
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Max Tree Depth')
    ax.set_zlabel('Accuracy')
    # Add a colorbar to the plot
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Accuracy')

    # Show the plot
    plt.title(f'3D Bar Plot of Accuracy Classifying {ticker} Changes vs Clusters and Tree Depth')
    plt.show()
    plt.savefig(f'hidden-stock-output/accuracy_plot_{ticker}.png')
    plt.close()


compressed_df = base_df.drop(columns=['num_clusters', 'max_depth'])
compressed_df = compressed_df.groupby('testing_ticker', as_index=False)['accuracy'].mean()
# Create a bar chart
plt.figure(figsize=(10, 7))
plt.bar(compressed_df['testing_ticker'], compressed_df['accuracy'])

# Add labels and title
plt.xlabel('Company')
plt.ylabel('Accuracy')
plt.title('Accuracy by Company')
plt.savefig(f'hidden-stock-output/average_accuracy_plot.png')