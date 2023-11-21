import pandas as pd
class DataPaths:
  @classmethod
  def supermarket_train_path(self):
    return 'drive/MyDrive/проекты/receipt_recom/supermarket_train.csv'
  
  @classmethod
  def supermarket_val_path(self):
    return 'drive/MyDrive/проекты/receipt_recom/supermarket_val.csv'
  @classmethod
  def supermarket_val_target_path(self):
    return 'drive/MyDrive/проекты/receipt_recom/supermarket_val_target.csv'

class TrainDataset(pd.DataFrame):
  def __init__(self, path):
    super().__init__(pd.read_csv(path, sep=';'))
    self['name'].fillna('', inplace=True)
    self.insert(1, 'label', 1)# = self.price.apply(lambda x:1)
  def save(self, path_out):
    self.to_csv(path_out, sep=';', index=False)

class TargetDataset(pd.DataFrame):
  def __init__(self, path):
    super().__init__(pd.read_csv(path, sep=';'))
    self.insert(1, 'label', 1)# = self.price.apply(lambda x:1)
  def save(self, path_out):
    self.to_csv(path_out, sep=';', index=False)