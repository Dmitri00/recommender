
class PairwiseFeaturizer:
  def __init__(self):
    pass
  def transform(self, logs):
    i2i = logs.merge(logs, on='receipt_id')
    i2i = i2i[~((i2i.item_id_x == i2i.item_id_y))]
    i2i = i2i.groupby('item_id_x item_id_y name_x name_y'.split()) \
      .agg(pair_cnt=('receipt_id', 'count')).reset_index()
    item_agg_stats = i2i.groupby('item_id_x'.split()) \
      .agg(item_cnt=('pair_cnt', 'count')) \
      .reset_index()
    i2i = i2i.merge(
        item_agg_stats.rename({'item_id_x':'item_id_x'},axis=1), 
        on='item_id_x') \
        .merge(
            item_agg_stats.rename({'item_id_x':'item_id_y'},axis=1), 
        on='item_id_y'
        )
    return i2i

class CandidateRevIndex:
  def __init__(self, topk=50, min_pair_cnt=10, max_iou=2):
    self.topk = topk
    self.min_pair_cnt = min_pair_cnt
    self.max_iou = max_iou
    pass
  
  def _calculate_pair_features(self):
    i2i = self.i2i
    self._feature_cols = []
    self.i2i['iou'] = i2i['pair_cnt'] / (i2i['item_cnt_x'] + i2i['item_cnt_y'] - i2i['pair_cnt'])
    self._feature_cols.append('iou')

  def _score_pairs(self):
    self.i2i['score'] = self.i2i['iou']
  
  def _topk(self):
    self.i2i = self.i2i.groupby('item_id_x').head(self.topk)

  
  def fit(self, pairs):
    self.i2i = pairs[pairs.pair_cnt>self.min_pair_cnt]
    

    self._calculate_pair_features()
    self.i2i[self.i2i['iou'] < self.max_iou]
    self._score_pairs()
    self._topk()
  
  def transform(self, user_signals):
    return user_signals.merge(self.i2i, on='r')
    
  def draft(self, user_signals):
    
    i2i = self.i2i
    user_signals = user_signals.rename({'item_id':'item_id_x', 'name':'name_x'}, axis=1)

    candidates = user_signals.merge(i2i, on='item_id_x name_x'.split()) \
      .rename({'item_id_y':'item_id', 'name_y':'name'}, axis=1) \
      ['receipt_id item_id name'.split()+self._feature_cols]

    return candidates



class TrendingIndex:
  def __init__(self, topk=100):
    self.topk = topk
    self.min_pair_cnt = min_pair_cnt
    self.max_iou = max_iou
    pass
  
  def _calculate_pair_features(self):
    i2i = self.i2i
    self._feature_cols = []
    self.i2i['iou'] = i2i['pair_cnt'] / (i2i['item_cnt_x'] + i2i['item_cnt_y'] - i2i['pair_cnt'])
    self._feature_cols.append('iou')

  def _score_pairs(self):
    self.i2i['score'] = self.i2i['iou']
  
  def _topk(self):
    self.i2i = self.i2i.groupby('item_id_x').head(self.topk)

  
  def fit(self, pairs):
    self.i2i = pairs[pairs.pair_cnt>self.min_pair_cnt]
    

    self._calculate_pair_features()
    self.i2i[self.i2i['iou'] < self.max_iou]
    self._score_pairs()
    self._topk()
  
  def transform(self, user_signals):
    return user_signals.merge(self.i2i, on='r')
    
  def draft(self, user_signals):
    
    i2i = self.i2i
    user_signals = user_signals.rename({'item_id':'item_id_x', 'name':'name_x'}, axis=1)

    candidates = user_signals.merge(i2i, on='item_id_x name_x'.split()) \
      .rename({'item_id_y':'item_id', 'name_y':'name'}, axis=1) \
      ['receipt_id item_id name'.split()+self._feature_cols]

    return candidates