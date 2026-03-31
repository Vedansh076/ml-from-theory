from .similarity import pearson_similarity
from .utils import mean_rating

def compute_user_similarities(data, target_user):
    sims = {}

    for user in data:
        if user == target_user:
            continue

        sim = pearson_similarity(data[target_user], data[user])
        sims[user] = sim

    return sims

def predict_rating(data, target_user, target_item, k=None):
    sims = compute_user_similarities(data, target_user)
    if k is not None:
        sims = dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)[:k])

    target_mean = mean_rating(data[target_user])

    numerator = 0
    denominator = 0

    for user, sim in sims.items():
        if target_item in data[user]:
            user_mean = mean_rating(data[user])

            numerator += sim * (data[user][target_item] - user_mean)
            denominator += abs(sim)
            
    if denominator == 0:
        return target_mean

    return target_mean + numerator / denominator