import csv
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from umap import UMAP
from collections import defaultdict
import matplotlib.pyplot as plt

# %% 1. Embeddings
# Read classmates and their responses from a CSV file
attendees_map = {}
with open("classmates.csv", newline="", encoding='utf-8') as csvfile:
    attendees = csv.reader(csvfile, delimiter=",", quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

# Create a dictionary to store embeddings for each person
person_embeddings = {
    attendees_map[paragraph]: embedding
    for paragraph, embedding in zip(paragraphs, embeddings)
}

# %% 2. Creating Visualization
# Reducing dimensionality of embedding data, scaling to coordinate domain/range
reducer = UMAP(random_state=42)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x, y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="3")

# Clean-up and Export
plt.axis("off")
plt.savefig("visualization.png")

# %% 3. Providing top matches
top_matches = {}
all_personal_pairs = defaultdict(list)
for person in attendees_map.values():
    for person1 in attendees_map.values():
        all_personal_pairs[person].append(
            [
                spatial.distance.cosine(
                    person_embeddings[person1], person_embeddings[person]
                ),
                person1,
            ]
        )

for person in attendees_map.values():
    top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[0])

print(top_matches)
