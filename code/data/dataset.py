from typing import *
from torch.utils import data

class Dataset(data.Dataset):
    """
    data를 model에 넣기위해 사용되는 dataset class
    """
    def __init__(self,description:str,title:str):
        self.description=description
        self.title=title
    
    def __getitem__(self,index:int)->Tuple[str,str]:
        return (self.description[index],self.title[index])
    
    def __len__(self)->int:
        return len(self.title)