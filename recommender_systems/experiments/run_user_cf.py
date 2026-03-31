import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from similarity_based.user_cf import predict_rating, compute_user_similarities

data = {
    "U1": {"A": 5, "B": 4},
    "U2": {"A": 5, "B": 4, "C": 5},
    "U3": {"A": 1, "B": 2, "C": 1}
}

sims = compute_user_similarities(data, "U1")
print("Similarities:", sims)

pred = predict_rating(data, "U1", "C", k=2)
print("Predicted rating for U1 on C:", pred)