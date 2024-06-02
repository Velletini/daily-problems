import json
#from compare_mt.rouge.rouge_scorer import RougeScorer
from multiprocessing import Pool
import os
from tqdm import tqdm
import argparse
from utils import cut_sent
from rouge import Rouge


rouge = Rouge()
#all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def collect_diverse_beam_data_cnewsum(args):
    split = args.split
    src_dir = args.src_dir
    tgt_dir = os.path.join(args.tgt_dir, split)
    cands = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.source"), encoding='utf-8') as src, open(os.path.join(src_dir, f"{split}.target"), encoding='utf-8') as tgt:
        with open(os.path.join(src_dir, f"{split}.out"), encoding='utf-8') as f_1:
            for x in  f_1:
                x = x.strip()
                cands.append(x)
                if len(cands) == args.cand_num:
                    src_line = src.readline()
                    src_line = src_line.strip()
                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip()
                    yield (src_line, tgt_line, cands, os.path.join(tgt_dir, f"{cnt}.json"), args.dataset)
                    cands = []
                    cnt += 1

def collect_diverse_beam_data(args):
    split = args.split
    src_dir = args.src_dir
    tgt_dir = os.path.join(args.tgt_dir, split)
    cands = []
    cands_untok = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.source.tokenized")) as src, open(os.path.join(src_dir, f"{split}.target.tokenized")) as tgt, open(os.path.join(src_dir, f"{split}.source")) as src_untok, open(os.path.join(src_dir, f"{split}.target")) as tgt_untok:
        with open(os.path.join(src_dir, f"{split}.out.tokenized")) as f_1, open(os.path.join(src_dir, f"{split}.out")) as f_2:
            for (x, y) in zip(f_1, f_2):
                x = x.strip()
                if args.lower:
                    x = x.lower()
                cands.append(x)
                y = y.strip()
                if args.lower:
                    y = y.lower()
                cands_untok.append(y)
                if len(cands) == args.cand_num:
                    src_line = src.readline()
                    src_line = src_line.strip()
                    if args.lower:
                        src_line = src_line.lower()
                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip()
                    if args.lower:
                        tgt_line = tgt_line.lower()
                    src_line_untok = src_untok.readline()
                    src_line_untok = src_line_untok.strip()
                    if args.lower:
                        src_line_untok = src_line_untok.lower()
                    tgt_line_untok = tgt_untok.readline()
                    tgt_line_untok = tgt_line_untok.strip()
                    if args.lower:
                        tgt_line_untok = tgt_line_untok.lower()
                    yield (src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok, os.path.join(tgt_dir, f"{cnt}.json"), args.dataset)
                    cands = []
                    cands_untok = []
                    cnt += 1


def build_diverse_beam(input):
    src_line, tgt_line, cands, tgt_dir, dataset = input
    cands = [cut_sent(x) for x in cands]
    abstract = cut_sent(tgt_line)
    #_abstract = "\n".join(abstract)
    article = cut_sent(src_line)
    

    def compute_rouge(hyp):
        try:
            score = rouge.get_scores(' '.join(list(tgt_line)), ' '.join(list(''.join(hyp))))
        except Exception as e:
            print(src_line, tgt_line, hyp)
            return 0
        #score = all_scorer.score(_abstract, "\n".join(hyp))
        return (score[0]["rouge-1"]['f'] + score[0]["rouge-2"]['f'] + score[0]["rouge-l"]['f']) / 3

    candidates = [(x, compute_rouge(x)) for x in cands]
    output = {
        "article": article, 
        "abstract": abstract,
        "candidates": candidates,
        }
    with open(tgt_dir, "w", encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)


def make_diverse_beam_data(args):
    with open(os.path.join(args.src_dir, f"{args.split}.source"), encoding='utf-8') as f:
        num = sum(1 for _ in f)
    data = collect_diverse_beam_data_cnewsum(args)
    with Pool(processes=80) as pool:
        for _ in tqdm(pool.imap_unordered(build_diverse_beam, data, chunksize=64), total=num):
            pass
    print("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing Parameter')
    parser.add_argument("--cand_num", type=int, default=16, help="Number of candidates")
    parser.add_argument("--src_dir", type=str,  default='data/CNewSum_v2/diverse', help="Source directory")
    parser.add_argument("--tgt_dir", type=str, default='data/CNewSum_v2/diverse', help="Target directory")
    parser.add_argument("--split", type=str, default='train', help="Dataset Split")
    parser.add_argument("--dataset", type=str, default="cnewsum", help="Dataset")
    #parser.add_argument("-l", "--lower", action="store_true", ,default=True, help="Lowercase")
    args = parser.parse_args()
    make_diverse_beam_data(args)
