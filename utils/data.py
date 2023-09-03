import torch
import pandas as pd
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, 
                df:pd.DataFrame,
                user_features:List, 
                item_features:List, 
                label:List,
                device:str):
        
        self.user_id = df[user_features].values
        self.movie_id = df[item_features].values
        self.rating = df[label].values
        self.device = device

    def __len__(self) -> int:
        return len(self.user_id)

    def __getitem__(self, idx):

        user_id = self.user_id[idx]
        movie_id = self.movie_id[idx]
        rating = self.rating[idx]

        return [torch.tensor(user_id).to(self.device),
                 torch.tensor(movie_id).to(self.device)], torch.tensor(rating).to(self.device)
    


def load_csv_dataset(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_train_test(df, 
                   user_features: List, 
                   item_features: List, 
                   label: List, 
                   device: str, 
                   batch_size: int=64, 
                   random_seed: int = 46):
                   
    train_data, test_data = train_test_split(df, 
                                             test_size=0.2,
                                             random_state=random_seed)

    train_dataloader = DataLoader(CustomDataset(
        train_data,
        user_features,
        item_features,
        label,
        device), batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(CustomDataset(
        test_data,
        user_features,
        item_features,
        label,
        device), batch_size=batch_size, shuffle=True)
    
    return test_dataloader, train_dataloader
    pass
def ml100k_dataset() -> pd.DataFrame:
    user = pd.read_table('ml-100k/u.user', header=None, delimiter='|', index_col=0)
    user = user.rename(columns={1: "age", 2: "gender", 3: "job", 4: "zip"})
    user['zip'] = user['zip'].map(lambda x: x[0] if x[0].isdigit() else 9)  # only first digit, change letter to 9, since it's minority(<10)
    user['gender'] = user['gender'].map(lambda x: 0 if x=='M' else 1)  # coding gender to binary
    user['job'] = user['job'].astype('category').cat.codes  # coding job to integer
    user = user.astype('int')
    user

    genre = pd.read_table('ml-100k/u.item', header=None, delimiter='|', encoding='latin-1', index_col=0).iloc[:, -19:]
    genre_name = pd.read_table('ml-100k/u.genre', header=None, delimiter='|')[0].values
    genre.columns = genre_name
    genre

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
    ratings_base

    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')
    items

    df = ratings_base.merge(user, left_on="user_id",right_index=True)
    df = df.merge(items, left_on="movie_id",right_on="movie id")
    return df