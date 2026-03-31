def mean_rating(user_ratings):
    return sum(user_ratings.values()) / len(user_ratings)

#check similarity only for common items between users
def common_items(u1, u2):
    return set(u1.keys()) & set(u2.keys())

def print_dict(d, title=""):
    if title:
        print(f"\n{title}")
    for k, v in d.items():
        print(f"{k}: {v:.3f}")