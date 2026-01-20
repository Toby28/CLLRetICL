import argparse
import os
import random
import numpy as np
from exp.exp_basic import Exp_Basic
from exp.exp_main import Exp_Main

fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Few-shot Learning LLM')

# data loader
parser.add_argument('--dataset', type=str, help='dataset, options:[sst2, cola, emotion, bbc];', default='bbc')
parser.add_argument('--save_results', type=int, required=False, default=1, help='save results into txt')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the results file')
parser.add_argument('--output_root_path', type=str, default='./results/', help='output root path of the results file')


parser.add_argument('--llm', type=str, default='gemini', help='LLM, options:[gemini, llama, mistral];')
parser.add_argument('--method', type=int, help='0: original; 1: contrastive; 2: labelaugment; 3:penalty',default=0)
parser.add_argument('--basemethod', type=str, help='icl,selfprompt,votek,simcse,knn,majorityvote,zicl',default='iclwo')
parser.add_argument('--NwayKshot', action='store_false', help='whether is NwayKshot', default=True)
parser.add_argument('--Nshot',type=int, help='the number of shots', default=3)
parser.add_argument('--zeroshot',action='store_false', help='whether is zeroshot', default=False)
parser.add_argument('--embeddingmethod', type=str, help='mpnet,bert,all-MiniLM-L6-v2',default='all-MiniLM-L6-v2')
parser.add_argument('--sim', type=str, help='cosine, euclidean',default='cosine')
parser.add_argument('--w1', type=float, help='',default=1.0)
parser.add_argument('--w2', type=float, help='',default=1.0)
parser.add_argument('--w3', type=float, help='0.3, 0.5 ,0.7, 1.0',default=1.0)
parser.add_argument('--api', type=str,default="LA-86e4e0dbe52d46178cdf38a0eb571bda48fb9c54b87f40c6bbb652e42808b1d7")
args = parser.parse_args()

print('Args in experiment:')
print(args)


# if args.is_training:
# for ii in range(args.itr):
    # setting record of experiments
setting = '{}_{}_{}_{}'.format(
    args.dataset,
    args.llm,
    args.method,
    args.basemethod)

print(setting)
exp = Exp_Main(args)  # set experiments
print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.run_results(setting)



