from torch.utils.data import Dataset
import json
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))


class CLSDataset(Dataset):

    def __init__(self,
                 data_path=os.path.join(os.path.dirname(cur_dir),
                                        "Datasets/CLS/"),
                 split="train",
                 device="cpu"):

        self.filename = os.path.join(data_path, "{}.json".format(split))

        with open(self.filename, encoding="utf-8") as f:
            self.data = json.load(f)

        self.padding_idx = None
        self.cls_idx = self.bos_idx = None
        self.sep_idx = self.eos_idx = None

        self.cls_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.pairs = []
        for article in self.data:
            content = article["Content"]
            for question in article['Questions']:
                q = question['Question']
                choices = question['Choices']
                label = self.cls_map[question['Answer']]
                self.pairs.append([content, q, choices, label])
        self.device = device
        self.split = split

    def __len__(self):
        return len(self.pairs)

    # @profile
    def __getitem__(self, index):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    # @profile
    def collate_fn(self, samples):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
