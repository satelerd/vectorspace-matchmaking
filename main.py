import csv
import umap
import warnings
import seaborn as sns
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Ignore innecesary package warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# Read data and their responses from a CSV file, replace data.csv with own link or file name
data_map = {}
with open('data.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(data)  # Skip the header row
    for row in data:
        name, paragraph = row
        data_map[paragraph] = name
print("Data loaded successfully.")

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(data_map.keys())
embeddings = model.encode(paragraphs)
print("Embeddings generated successfully.")
    
# Create a dictionary to store embeddings for each person
person_embeddings = {data_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

# Reducing dimensionality of embedding data, scaling to coordinate domain/range
reducer = umap.UMAP()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)
print("Dimensionality reduction and scaling completed successfully.")

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x,y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="15", color="black")  # Increased font size to 12 and set color to white

# Clean-up and Export
plt.axis('off')
plt.savefig('visualization.png', dpi=800)
print("Visualization saved successfully.")