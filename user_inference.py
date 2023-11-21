from sklearn.linear_model import LogisticRegression
class UserFeatureAgg:
  def __init__(self, groupby_col, col_set, agg_func_set, features_col):
    self.col_set = col_set
    self.agg_func_set = agg_func_set
    self.groupby_col = groupby_col
    self.features_col = features_col

  def fit(self):
    self.agg_exprs = describe_agg_functions(self.col_set, self.agg_func_set)
  
  def transform(self, candidates):
    candidates = candidates.groupby(self.groupby_col).agg(**self.agg_exprs)
    candidates = candidates.reset_index()
    candidates[self.features_col] = candidates.apply(lambda row:
                                                     np.array([row[col] for col in self.agg_exprs.keys()]), axis=1)
    return candidates


from itertools import product
def describe_agg_functions(col_set, agg_funcs):
  aggrs = {}
  for col, agg_name in product(col_set, agg_funcs):
    new_col_name = f'{agg_name}_{col}'
    aggrs[new_col_name] = (col, agg_name)
  return aggrs



class TopK:
  def _topk(self, df, col):
    return df.groupby(col).head(self.topk)

class UserRecommender(TopK):
  def __init__(self, candidate_generator, feature_aggregator, scorer, topk=30):
    self.candidate_generator = candidate_generator
    self.feature_aggregator = feature_aggregator
    self.scorer = scorer
    self.topk = topk
  
  
  def transform(self, user_signals):

    candidates = self.candidate_generator.draft(user_signals)

    candidates_aggregated = self.feature_aggregator.transform(candidates)

    candidates_scored = self.scorer.transform(candidates_aggregated)

    print(f'Len of candidates before top{self.topk}:{len(candidates_scored)}')
    top_candidates = self._topk(candidates_scored, 'receipt_id')
    print(f'Len of candidates after top{self.topk}:{len(top_candidates)}')


    return top_candidates

      
class Scorer:
  def __init__(self, features_col, score_col, label_col):
    self.score_col = score_col
    self.features_col = features_col
    self.label_col = label_col
  def _score_func(self, features):
    return np.sum(features)
  def fit(self, train):
    self._fit(train[self.features_col].tolist(), train[self.label_col].tolist())
  def transform(self, candidates):
    scores = self._score_func(candidates[self.features_col].tolist())
    print(scores)
    candidates[self.score_col] = scores
    return candidates
  
class LogRegScorer(Scorer):
  def __init__(self, features_col, score_col, label_col):
    super().__init__(features_col, score_col, label_col)
    self.model = LogisticRegression()
  def _fit(self, x_train, y_train):
    self.model.fit(x_train, y_train)
  def _score_func(self, features):
    return self.model.predict_proba(features)[:,1]