import csv
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# %% 1. Generate new Embeddings
# Read classmates and their responses from a CSV file
attendees_map = {}
with open("classmates.csv", newline="", encoding="utf-8") as csvfile:
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
with open("person_embeddings.pkl", "rb") as f:
    old_person_embeddings = pickle.load(f)

changed_classmates = ["Pawan Lingras", "Sachit Jain", "Greg Kirczenow"]
# benchmark:  I did not update my info, so should have a similarity score of 1 (equal)
benchmark = cosine_similarity(
    [old_person_embeddings["Somto Muotoe"]], [person_embeddings["Somto Muotoe"]]
)
# array([[1.]], dtype=float32)

similarity_scores = {}
for classmate in changed_classmates:
    similarity_scores[classmate] = cosine_similarity(
        [old_person_embeddings[classmate]], [person_embeddings[classmate]]
    )

print(similarity_scores)

# {
# 'Pawan Lingras': array([[0.9550047]], dtype=float32),
# 'Sachit Jain': array([[0.8030659]], dtype=float32),
# 'Greg Kirczenow': array([[0.48249182]], dtype=float32)
# }
