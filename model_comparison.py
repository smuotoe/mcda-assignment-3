import csv
import pickle

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# Read classmates and their responses from a CSV file
attendees_map = {}
with open("classmates.csv", newline="", encoding="utf-8") as csvfile:
    attendees = csv.reader(csvfile, delimiter=",", quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# List of models to compare
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

# Dictionary to store person embeddings for each model
person_embeddings_models = {}

for model_name in models:
    # Generate sentence embeddings
    model = SentenceTransformer(model_name)
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)

    # Create a dictionary to store embeddings for each person
    person_embeddings = {
        attendees_map[paragraph]: embedding
        for paragraph, embedding in zip(paragraphs, embeddings)
    }

    # Save embeddings as pickle object.
    with open(f"person_embeddings_{model_name.split('/')[-1]}.pkl", "wb") as f:
        pickle.dump(person_embeddings, f)

    person_embeddings_models[model_name] = person_embeddings

# Compute cosine distances between myself and each classmate for both models
my_name = "Somto Muotoe"
distances_models = {
    model_name: sorted(
        [
            (name, cosine(person_embeddings[my_name], embedding))
            for name, embedding in person_embeddings.items()
        ],
        key=lambda x: x[1],
    )
    for model_name, person_embeddings in person_embeddings_models.items()
}

# Compute the rank correlation between the distances from the two models
ranks_minilm = [name for name, _ in distances_models[models[0]]]
ranks_mpnet = [name for name, _ in distances_models[models[1]]]
rank_correlation, _ = spearmanr(ranks_minilm, ranks_mpnet)

print(f"Spearman's rank correlation: {rank_correlation}")
# Spearman's rank correlation: 0.02714285714285714

# >>> ranks_minilm ['Somto Muotoe', 'Pranay Malusare', "D'Shon Henry", 'Princeton Dcunha', 'Venkata Sujay Kumar Vemuri',
# 'Ajay Jain', 'Carmen Leung', 'Zaid', 'Abhilash Sibi', 'Jerry Caleb', 'Nghia Phan', 'Rishabh Khevaria',
# 'Abhishek Vijayakumar ', 'Julius Sun', 'Sudeep Raj Badal', 'Samuel Ebong', 'Shweta Dalal', 'Tejasvi Bhutiyal',
# 'Aditya Chaudhari', 'Arpan Patel', 'Vrushali Prajapati', 'Lilian Guo', 'Raoof Naushad', 'Kritika Koirala',
# 'Royston Furtado', 'Rakshit Gupta', 'Greg Kirczenow', 'Sameer Patel', 'Weilin Wang', 'Sylvester Terdoo',
# 'Bhavy Doshi', 'Aravind Gopi', 'Mehul Patel', 'Sharlene Karina Wadhwa', 'Nikita Neveditsin', 'Francis Kuzhippallil
# ', 'Sachit Jain', 'Pawan Lingras', 'Akash Pandey', 'Rashad Ahmed', 'Neeyati Mehta', 'Roy Jasper',
# 'Shiney Prabhakar', 'Piyush Priyam ', 'Deepakk Vignesh Jayamohan', 'Subhiksha Ramasubramanian ', 'Kin Wa Chan',
# 'Hritik Arora', 'Andy Wang']

# >>> ranks_mpnet ['Somto Muotoe', "D'Shon Henry", 'Pranay Malusare', 'Jerry Caleb',
# 'Abhilash Sibi', 'Julius Sun', 'Venkata Sujay Kumar Vemuri', 'Aditya Chaudhari', 'Nghia Phan', 'Zaid',
# 'Greg Kirczenow', 'Princeton Dcunha', 'Carmen Leung', 'Shweta Dalal', 'Sudeep Raj Badal', 'Samuel Ebong',
# 'Arpan Patel', 'Lilian Guo', 'Ajay Jain', 'Sachit Jain', 'Aravind Gopi', 'Sameer Patel', 'Pawan Lingras',
# 'Weilin Wang', 'Sylvester Terdoo', 'Tejasvi Bhutiyal', 'Abhishek Vijayakumar ', 'Mehul Patel', 'Rakshit Gupta',
# 'Kritika Koirala', 'Roy Jasper', 'Nikita Neveditsin', 'Akash Pandey', 'Bhavy Doshi', 'Shiney Prabhakar',
# 'Rishabh Khevaria', 'Raoof Naushad', 'Neeyati Mehta', 'Francis Kuzhippallil ', 'Vrushali Prajapati',
# 'Rashad Ahmed', 'Royston Furtado', 'Sharlene Karina Wadhwa', 'Subhiksha Ramasubramanian ', 'Deepakk Vignesh
# Jayamohan', 'Piyush Priyam ', 'Andy Wang', 'Kin Wa Chan', 'Hritik Arora']
