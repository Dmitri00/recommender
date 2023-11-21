from evaluator import MergeLabelsScores
from datasets import TrainDataset, TargetDataset, DataPaths
from user_inference import UserFeatureAgg, UserRecommender
from candidate_generators import PairwiseFeaturizer, CandidateRevIndex

def main():
  train = TrainDataset(DataPaths.supermarket_train_path())
  val = TrainDataset(DataPaths.supermarket_val_path())
  val_target = TargetDataset(DataPaths.supermarket_val_target_path())
  rank_labler = MergeLabelsScores()
  user_feature_agg_train = UserFeatureAgg('receipt_id item_id label'.split(), 'iou'.split(), 'max mean'.split(), 'features')
  user_feature_agg_train.fit()
  user_feature_agg_inference = UserFeatureAgg('receipt_id item_id'.split(), 'iou'.split(), 'max mean'.split(), 'features')
  user_feature_agg_inference.fit()

  # Candidate Generators:
  pairs = PairwiseFeaturizer().transform(train)
  revidx = CandidateRevIndex(max_iou=11)
  revidx.fit(pairs)
  ### Top level Reranker Train
  train_featured = prepare_scorer_train(train, [revidx], user_feature_agg_train, negative_frac=0.03)
  logreg_scorer = LogRegScorer('features', 'score', 'label')
  logreg_scorer.fit(train_featured)
  user_feature_agg.fit()

  ### User Ranker Framework
  user_model = UserRecommender(revidx, user_feature_agg_inference, logreg_scorer, topk=30)
  candidates = user_model.transform(val)
  y_true = val_target['receipt_id item_id label'.split()]
  y_pred = candidates['receipt_id item_id score'.split()]
  metric = MergeLabelsScores.score(y_true, y_pred)

if __name__ == '__main__':
  main()