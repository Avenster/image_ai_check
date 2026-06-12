import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_disagreement(confidences: list) -> dict:
    # Jensen-Shannon distances between all pairs (treat each as binary distribution)
    pairs_js = []
    for i in range(len(confidences)):
        for j in range(i+1, len(confidences)):
            p = np.array([1-confidences[i], confidences[i]])
            q = np.array([1-confidences[j], confidences[j]])
            js = jensenshannon(p, q)
            pairs_js.append((i, j, js))
    mean_js = float(np.mean([js for _,_,js in pairs_js])) if pairs_js else 0.0
    # Also compute standard deviation of confidences
    std_conf = float(np.std(confidences))
    return {
        'mean_js_divergence': mean_js,
        'std_confidence': std_conf,
        'pairwise_js': pairs_js
    }
