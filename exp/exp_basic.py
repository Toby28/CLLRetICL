import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        if self.args.zeroshot==False:
            self.setting = '{}_{}_{}_{}_{}_{}'.format(
                args.dataset,
                args.embeddingmethod,
                args.llm,
                args.basemethod,
                args.Nshot,
                args.sim
            )
        else:
            self.setting = '{}_{}_zeroshot'.format(
                args.dataset,
                args.llm)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def run_results(self):
        pass
