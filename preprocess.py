import json
from compare_mt.rouge.rouge_scorer import RougeScorer
from multiprocessing import Pool
import os
import random
from itertools import combinations
from functools import partial
import re
import nltk
import numpy as np

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

systems = ["match", "bart"]
scorer = RougeScorer(['rouge1'], use_stemmer=True)
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def make_trigrams(x):
    x = x.split(" ")
    trigrams = set()
    for i in range(len(x) - 2):
        trigrams.add(" ".join(x[i:i+3]))
    return trigrams

def exist_trigram(s, c):
    for x in s:
        a = make_trigrams(x)
        b = make_trigrams(c)
        if not a.isdisjoint(b):
            return True
    return False

def make_rank(fname):
    src_path = "./CNNDM/two_mat/train"
    tgt_path = "./CNNDM/rank_mat/train"
    with open(os.path.join(src_path, fname)) as f:
        data = json.load(f)
    match = data["candidates"][0][0]
    bart = data["candidates"][1][0]
    match, bart = match[:4], bart[:4]
    sents = match + bart
    sent_id = [_ for _ in range(len(sents))]
    indices = list(combinations(sent_id, 3))
    # indices += list(combinations(sent_id, 3))
    if len(sent_id) < 2:
        indices = [sent_id]
    cands = []
    scores = []
    ref = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(ref, "\n".join(hyp))
        score = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
        return score

    for ids in indices:
        cand = []
        flag = False
        ids = list(ids)
        ids.sort()
        for id in ids:
            if exist_trigram(cand, sents[id]):
                flag = True
                break
            cand.append(sents[id])
        if not flag:
            scores.append(compute_rouge(cand))
            cands.append(ids)
    if len(cands) == 0:
        cands.append([_ for _ in range(len(match))])
        cands.append([_ for _ in range(len(match), len(bart))])
        scores.append(compute_rouge(match))
        scores.append(compute_rouge(bart))
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    with open(os.path.join(tgt_path, fname), "w") as f:
        json.dump(output, f)
    
def make_rank_data():
    unfinish_files = os.listdir("./CNNDM/two_mat/train")
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(make_rank, unfinish_files, chunksize=64))
    print("finish")

def split(x, lower=True):
    if lower:
        x = x.lower()
    x = x.strip().split(".")
    result = []
    for s in x:
        if len(s.strip()) > 0: 
            result.append(s.strip() + " .")
    return result

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)



def collect_xsum_data(split):
    base_dir = f"./xsum/base/{split}"
    tgt_dir = f"./xsum/two/{split}"
    with open(f"./data/xsum/xsum.{split}.out.tokenized") as f:
        bart = [x.strip().lower() for x in f]
    with open(f"./data/xsum/{split}.match.idx") as f:
        idx = json.load(f)
    with open(f"./data/xsum.{split}.match.jsonl") as f:
        for i, match in enumerate(f):
            yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[idx[i]], match)

def build_xsum(input):
    src_dir, tgt_dir, bart, match = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    bart = sent_detector.tokenize(bart)
    match = json.loads(match)
    candidates = [(match, compute_rouge(match)), (bart, compute_rouge(bart))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_data(split):
    data = collect_xsum_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_xsum, data, chunksize=64))
    print("finish")

def collect_other_data(split):
    base_dir = f"./pubmed/base/{split}"
    tgt_dir = f"./pubmed/two/{split}"
    with open(f"./data/pubmed/pubmed.{split}.out.tokenized") as f:
        bart = [x.strip().lower() for x in f]
    with open(f"./data/pubmed.{split}.match.jsonl") as f:
        for i, match in enumerate(f):
            yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[i], match)

def build_other(input):
    src_dir, tgt_dir, bart, match = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    bart = sent_detector.tokenize(bart)
    match = json.loads(match)
    candidates = [(match, compute_rouge(match)), (bart, compute_rouge(bart))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_other_data(split):
    data = collect_other_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_other, data, chunksize=64))
    print("finish")

def gather_beam_data(fdir):
    sents = []
    with open(fdir) as f:
        for x in f:
            x = x.strip().lower()
            sents.append(x)
            if len(sents) == 4:
                yield sents
                sents = []


def collect_xsum_beam_data(split):
    base_dir = f"./data/{split}_wikihow.jsonl"
    tgt_dir = f"./wikihow/beam/{split}"
    beam_data = gather_beam_data(f"./data/wikihow/wikihow.{split}.beam.out.tokenized")
    with open(base_dir) as f:
        for (i, (beam, line)) in enumerate(zip(beam_data, f)):
            yield (line, os.path.join(tgt_dir, f"{i}.json"), beam)

def build_beam_xsum(input):
    src_dir, tgt_dir, bart = input
    data = json.loads(src_dir)
    abstract = "\n".join(data["summary"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    bart = [sent_detector.tokenize(x) for x in bart]
    candidates = [(x, compute_rouge(x)) for x in bart]
    output = {
        "article": data["text"], 
        "abstract": data["summary"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_beam_data(split):
    data = collect_xsum_beam_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_beam_xsum, data, chunksize=64))
    print("finish")

def collect_xsum_new_data(split):
    base_dir = f"./xsum/base/{split}"
    tgt_dir = f"./xsum/new/{split}"
    bart = []
    pegasus = []
    for i in range(1, 4):
        with open(f"./data/xsum/{split}.bart.out.tokenized") as fb, open(f"./data/xsum/{split}.pegasus.out.tokenized") as fp: 
            for (x, y) in zip(fb, fp):
                bart.append(x.strip().lower())
                pegasus.append(y.strip().lower())
    nums = len(os.listdir(base_dir))
    with open(f"./data/xsum/{split}.match.idx") as f:
        idx = json.load(f)
    for i in range(nums):
        yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[idx[i]], pegasus[idx[i]])

def build_new_xsum(input):
    src_dir, tgt_dir, bart, pegasus = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    bart = [bart]
    pegasus = [pegasus]
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    candidates = [(bart, compute_rouge(bart)), (pegasus, compute_rouge(pegasus))] # bart, pegasus
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_new_data(split):
    data = collect_xsum_new_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_new_xsum, data, chunksize=64))
    print("finish")

def collect_oracle_data(split):
    src_dir = f"./pubmed/beam/{split}"
    tgt_dir = f"./pubmed/oracle/{split}"
    num = len(os.listdir(src_dir))
    for i in range(num):
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"))

def build_oracle(input):
    src_dir, tgt_dir = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    sent_scores = [compute_rouge([x]) for x in data["article"]]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:7]
    sents = data["article"]
    # indices = list(combinations(sent_id, 3)) + list(combinations(sent_id, 4)) + list(combinations(sent_id, 5))
    indices = list(combinations(sent_id, 6))
    if len(sent_id) < 6:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_oracle_data(split):
    data = collect_oracle_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_oracle, data, chunksize=64))
    print("finish")

def select_samples(x, max_num=30):
    if len(x) == 1:
        return [0]
    if len(x) == 2:
        return [0, 1]
    num = len(x)
    ids = np.random.choice(num - 2, size=min(num - 2, max_num - 2), replace=False) + 1
    ids = ids.tolist()
    ids.sort()
    ids.append(-1)
    ids = [0] + ids
    return ids 

def select_oracle_data(split):
    src_dir = f"./CNNDM/oracle/{split}"
    tgt_dir = f"./CNNDM/oracle_30/{split}"
    num = len(os.listdir(src_dir))
    for i in range(num):
        if i % 1000 == 0:
            print(i)
        with open(os.path.join(src_dir, f"{i}.json")) as f:
            data = json.load(f)
        ids = select_samples(data["scores"])
        data["scores"] = [data["scores"][i] for i in ids]
        data["indices"] = [data["indices"][i] for i in ids]
        with open(os.path.join(tgt_dir, f"{i}.json"), "w") as f:
            json.dump(data, f)
        
def collect_combine_data(split):
    bart_dir = f"./xsum/beam_bart/{split}"
    pegasus_dir = f"./xsum/beam_pegasus/{split}"
    tgt_dir = f"./xsum/combine/{split}"
    num = len(os.listdir(bart_dir))
    for i in range(num):
        yield (os.path.join(bart_dir, f"{i}.json"), os.path.join(pegasus_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"))

def build_combine(input):
    bart_dir, pegasus_dir, tgt_dir = input
    with open(bart_dir) as f:
        data = json.load(f)
    with open(pegasus_dir) as f:
        pegasus_data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    sent_scores = [compute_rouge([x]) for x in data["article"]]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:5]
    sents = data["article"]
    indices = list(combinations(sent_id, 1)) + list(combinations(sent_id, 2))
    if len(sent_id) < 2:
        indices = [sent_id]
    scores = [x[1] for x in data["candidates"]] + [x[1] for x in pegasus_data["candidates"]]
    candidates = [x[0] for x in data["candidates"]] + [x[0] for x in pegasus_data["candidates"]]
    cands = []
    _scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        _scores.append(compute_rouge(cand))
        cands.append(cand)
    tmp = zip(cands, _scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    tmp = tmp[:6]
    cands = [y[0] for y in tmp]
    _scores = [y[1] for y in tmp]
    candidates.extend(cands)
    scores.extend(_scores)
    candidates = [(candidates[i], scores[i]) for i in range(len(scores))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_combine_data(split):
    data = collect_combine_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_combine, data, chunksize=64))
    print("finish")

def collect_diverse_data(split):
    src_dir = f"./wikihow/beam/{split}"
    tgt_dir = f"./wikihow/diverse/{split}"
    cands = []
    sents = []
    with open(f"./data/wikihow/{split}.diverse.out.tokenized") as f:
        for x in f:
            x = x.strip().lower()
            sents.append(x)
            if len(sents) == 16:
                cands.append(sents)
                sents = []
    nums = len(cands)
    print(nums)
    for i in range(nums):
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), cands[i])

def build_diverse(input):
    src_dir, tgt_dir, cands = input
    cands = list(set(cands))
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    cands = [(x, compute_rouge([x])) for x in cands]
    cands = sorted(cands, key=lambda x: x[1], reverse=True)
    indices = [[i] for i in range(len(cands))]
    scores = [x[1] for x in cands]
    cand_sents = [x[0] for x in cands]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": indices,
        "scores": scores,
        "cand_sents": cand_sents
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_diverse_data(split):
    data = collect_diverse_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_diverse, data, chunksize=64))
    print("finish")

def collect_pretrain_data(split):
    src_dir = f"./CNNDM/pre/{split}"
    tgt_dir = f"./CNNDM/pretrain/{split}"
    # num = len(os.listdir(src_dir))
    files = os.listdir(src_dir)
    cnt = 0
    for (i, x) in enumerate(files):
        with open(os.path.join(src_dir, f"{i}.json")) as f:
            data = json.load(f)
        if len(data["article"]) < 7:
            continue
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{cnt}.json"))
        cnt += 1
    print(cnt)

def build_pretrain(input):
    src_dir, tgt_dir = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["article"][:3])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    cand_sents = data["article"][3:]
    sent_scores = [compute_rouge([x]) for x in cand_sents]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:5]
    indices = list(combinations(sent_id, 2)) + list(combinations(sent_id, 3))
    # indices = list(combinations(sent_id, 6))
    if len(sent_id) < 3:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [cand_sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": cand_sents, 
        "abstract": data["article"][:3],
        "indices": cands,
        "scores": scores,
        "cand_sents": cand_sents,
        "real_abs": data["abstract"]
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_pretrain_data(split):
    data = collect_pretrain_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_pretrain, data, chunksize=64))
    print("finish")



    

