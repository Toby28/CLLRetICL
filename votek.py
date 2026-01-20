##########
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def fast_votek(embeddings,select_num,k,vote_file=None):
    n = len(embeddings)
    # if vote_file is not None and os.path.isfile(vote_file):
        # with open(vote_file) as f:
            # vote_stat = json.load(f)
    # else:
    # bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        # bar.update(1)
    if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices