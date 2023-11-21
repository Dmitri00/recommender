from sklearn.metrics import ndcg_score, dcg_score,coverage_error,label_ranking_average_precision_score

class MergeLabelsScores:
  def __init__(self, merge_col='receipt_id item_id'.split(), label_col='label', score_col='score'):
    self.merge_col = merge_col
    self.label_col = label_col
    self.score_col = score_col
    pass
  
  def transform(self, targets, recoms):
    merged = targets.merge(recoms, on=self.merge_col, how='right', suffixes='_true _pred'.split())

    merged[self.label_col].fillna(0, inplace=True)
    merged = merged.sort_values(self.score_col, ascending=False)
    merged = merged.groupby('receipt_id') \
      .agg(labels=(self.label_col, list),
           recoms=(self.score_col, list)) \
      .reset_index()
    return merged

  def score(self, y_true, y_pred, k=10):
    true_vs_recom = self.transform(
        y_true,
        y_pred)
    max_recom_size  = true_vs_recom.labels.apply(len).max()
    # этот ndcg не умеет работать с массивами разной длины, поэтому пока ставлю
    # костыль на мой размер драфта
    data = true_vs_recom[true_vs_recom.labels.apply(len)==30]
    metric = ndcg_score(data['labels'].tolist(), data['recoms'].tolist(), k=k)
    return metric