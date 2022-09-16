import os
import pandas as pd
from sklearn import preprocessing
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import CFG

def main():
    train_df = pd.read_csv('train.csv')

    submission = pd.read_csv('sample_submission.csv')
    train_dir = 'train_images'

    train_df['path_jpeg'] = train_df.apply(lambda row: os.path.join(train_dir , row['label'] , row['image_id']), axis=1)

    le = preprocessing.LabelEncoder()
    le.fit(train_df['label'])
    train_df['label'] = le.transform(train_df['label'])

    le.fit(train_df['variety'])
    train_df['variety'] = le.transform(train_df['variety'])

    # ====================================================
    # CV schem
    # ====================================================
    skf = MultilabelStratifiedKFold(n_splits=CFG.nfolds, shuffle=True, random_state=CFG.seed)
    for fold, (trn_idx, vld_idx) in enumerate(skf.split(train_df, train_df[['label', 'age', 'variety']])):
        train_df.loc[vld_idx, "fold"] = int(fold)
    train_df["fold"] = train_df["fold"].astype(int)
    train_df.to_csv('train_df')

if __name__ == "__main__":
    main()