import math
from .utils import mean_rating, common_items

def pearson_similarity(u1, u2):
    items = common_items(u1, u2) #reduce dimensions: only overlaps matter
    if len(items) == 0:
        return 0
    mean1 = mean_rating({i: u1[i] for i in items})
    mean2 = mean_rating({i: u2[i] for i in items})

    numerator = 0
    for i in items:
        numerator += (u1[i] - mean1) * (u2[i] - mean2)
    
    denom1 = math.sqrt(sum((u1[i] - mean1)**2 for i in items))
    denom2 = math.sqrt(sum((u2[i] - mean2)**2 for i in items))
    if denom1 == 0 or denom2 == 0:
        return 0
    
    return numerator / (denom1 * denom2)