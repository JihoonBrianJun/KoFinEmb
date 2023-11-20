import pandas as pd
import numpy as np
from argparse import ArgumentParser

def main():
    all = pd.read_csv(args.thread_path)
    returns = pd.read_csv(args.return_path)
    
    all['day'] = all['datetime'].apply(lambda x: x.split(" ")[0].replace(".","-")).astype(str)
    all['code'] = all['code'].astype(str)
    returns['day'] = returns['day'].astype(str)
    returns['code'] = returns['code'].astype(str)
    
    all_returns = pd.merge(all, returns, on=["day", "code"], how="left").drop_duplicates(subset=["day","code","title"])
    all_returns['label'] = all_returns['return'].apply(lambda x: "[상승]" if x>=0 else "[하락]")
    all_returns['title'] = all_returns['title'].astype(str).apply(lambda x: x.split("[")[0] if "[" in x else x)
    all_returns['title_len'] = all_returns['title'].apply(lambda x: len(x))
    all_returns = all_returns[all_returns['title_len']>=args.title_min_len]
    all_returns = all_returns[['title','label']]
    
    all_returns_positive = all_returns[all_returns['label']=='[상승]']
    all_returns_negative = all_returns[all_returns['label']=='[하락]']
    
    print(all_returns.shape)
    print(f"상승 개수: {all_returns_positive.shape}")
    print(f"하락 개수: {all_returns_negative.shape}")
    
    all_returns = pd.concat([all_returns_positive.sample(args.instance_num//2, replace=False),
                             all_returns_negative.sample(args.instance_num//2, replace=False)],
                            axis=0)

    print(all_returns.head())
    print(all_returns.shape)
    
    train_instance_num = int(all_returns.shape[0] * args.train_ratio)
    train_idx = np.random.choice(all_returns.shape[0], train_instance_num, replace=False)
    eval_idx = np.array(list(set(list(np.arange(all_returns.shape[0]))).difference(set(list(train_idx)))))
    
    all_returns_train = all_returns.iloc[train_idx]
    all_returns_eval = all_returns.iloc[eval_idx]
    
    all_returns_train.to_csv(args.train_out_path, index=False)
    all_returns_eval.to_csv(args.eval_out_path, index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--thread_path", type=str, default="dataset/all.csv")
    parser.add_argument("--return_path", type=str, default="dataset/return.csv")
    parser.add_argument("--title_min_len", type=int, default=20)
    parser.add_argument("--instance_num", type=int, default=30000)
    parser.add_argument("--train_out_path", type=str, default="dataset/train.csv")
    parser.add_argument("--eval_out_path", type=str, default="dataset/eval.csv")
    parser.add_argument("--train_ratio", type=float, default=0.99)
    args = parser.parse_args() 
    main()