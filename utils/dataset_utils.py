import copy
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Sequence
import pandas as pd
import torch
from torch.utils.data import Dataset

class DatasetGenerator(Dataset):
    def __init__(self, tokenizer, config, dataset_type):
        super(DatasetGenerator, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100
        
        if dataset_type == "train":
            data_path = config.train_data_path
        else:
            data_path = config.eval_data_path
        
        instruction = """주식 종목 토론방에 [제목]의 글이 올라온 다음 날, 주가는 어떻게 되었을까?
        [상승] 혹은 [하락] 중 하나로 답하여라. 
        
        [제목]: 장마감 이후 올라온 공시 자료를 보니, 분기 실적이 예상치를 크게 밑도네요ㅠ
        [정답]: [하락]
        
        [제목]: 간밤에 미국 연준의 깜짝 금리 인하 결정에 힘입어 미국 증시 폭등 중이네요. 내일 한국 증시도 기대해 봅니다!
        [정답]: [상승]
        """
        
        df = pd.read_csv(data_path)
        df["title_with_instruction"] = df["title"].apply(lambda x: f"{instruction}\n\n[제목]: {x}\n[정답]: ")
        # df["classify_label"] = df["label"].apply(lambda x: f"{x}{self.tokenizer.eos_token}")
        df["classify_label"] = df["label"]
        
        input_list = df["title_with_instruction"].tolist()
        if dataset_type == "train":
            target_list = df["classify_label"].tolist()
        else:
            target_list = None

        print("Tokenizing inputs... This may take some time...")
        if target_list != None:
            example_list = [input+target for input, target in zip(input_list, target_list)]
        else:
            example_list = input_list
        
        example_tokenized, input_tokenized = [self.tokenize(data_list) 
                                            for data_list in (example_list, input_list)]
        input_ids = example_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        print("Tokenizing All done!")
        
        for i in tqdm(range(len(labels))):
            input_len = input_tokenized["input_ids_lens"][i] 
            labels[i][:input_len] = self.IGNORE_INDEX
        
        data_dict = dict(input_ids=input_ids, labels=labels)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.dataset_type = dataset_type
        print(f"Example dataset..\nInput: {input_list[0]}\ninput_ids: {self.input_ids[0]}\nlabels: {self.labels[0]}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], dataset_type=self.dataset_type)
    
    def tokenize(self, data_list):
        tokenized_list = []
        for data in data_list:
            tokenized_list.append(self.tokenizer(data,
                                                 return_tensors="pt",
                                                 padding="longest",
                                                 max_length = self.config.max_length,
                                                 truncation=True))
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
                          for tokenized in tokenized_list]
        
        return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


@dataclass
class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, dataset_type = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "dataset_type"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=self.tokenizer.pad_token_id)
        
        return dict(input_ids=input_ids, 
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))