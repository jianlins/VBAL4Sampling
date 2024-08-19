from quicksectx import IntervalTree, Interval
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite import CRF
import pandas as pd
import numpy as np
from spacy.tokens import Doc
from typing import List
from loguru import logger
import random

def spans_to_bio(doc:Doc, anno_types:List[str], abbr:bool=False)->str:
  """
  Converts spans in a spaCy Doc object to a BIO-formatted string, with an option
  to abbreviate the entity labels. It adds an empty line between sentences to improve
  readability.

  Parameters:
  - doc (Doc): The spaCy Doc object containing the text and its annotations, including
                entities and sentence boundaries.
  - anno_types (List[str]): A list of annotation types to include in the output. These
                            types should correspond to the keys in `doc.spans`.
  - abbr (bool, optional): If True, entity labels are abbreviated to their initials.
                            Defaults to True.

  Returns:
  - str: A string where each token is followed by its BIO tag (with the entity label if applicable),
          formatted as "token B-entity" or "token I-entity" for tokens within entities, and
          "token O" for tokens outside any entities. Sentences are separated by an empty line.
  """
  # Initialize a dictionary to hold BIO tags for each token index
  bio_tags = {token.i: 'O' for token in doc}  # Default to 'O' for outside any entity

  # Preprocess spans to assign BIO tags
  for anno_type, spans in doc.spans.items():
    if anno_type not in anno_types:
        continue
    if len(spans)==0:
        continue
    for span in spans:
        if span:  # Check if span is not empty
          label=span.label_
          if label not in anno_types:
            continue
          if abbr:
            label=''.join([w[0] for w in label.split('_')])
          bio_tags[span.start] = f"B-{label}"  # Begin tag for the first token in the span
          for token in span[1:]:  # Inside tags for the rest of the tokens in the span
            bio_tags[token.i] = f"I-{label}"

  # Generate BIO format string
  bio_text = []
  bio_data={'sentence_id':[],'doc_name':[], 'token':[],'label':[]}
  for s,sent in enumerate(doc.sents):
    for i,token in enumerate(sent):
      # trim the whitespaces on both sides of a sentence
      if (i==0 or i==len(sent)-1) and str(token).strip()=='':
        bio_text.append('')
        continue
      elif str(token).strip()=='':
        # clean up extra whitespaces within a sentence.
        bio_text.append(f' \t{bio_tags[token.i]}')
        bio_data['label'].append(bio_tags[token.i])
      else:
        bio_text.append(f"{token.text} {bio_tags[token.i]}")
        bio_data['label'].append(bio_tags[token.i])
      bio_data['doc_name'].append(doc._.doc_name)
      bio_data['token'].append(token)
      bio_data['sentence_id'].append(s)
    bio_text.append('')  # Empty line between sentences
  return '\n'.join(bio_text), pd.DataFrame(bio_data)

# We will focus on two types of concepts here
def convert_docs(docs:List[Doc], anno_types=['FAM_COLON_CA','COLON_CA']):
  all_conll=[]
  offset=0
  dfs=[]
  for d in docs:
    data, df=spans_to_bio(d, anno_types=anno_types)
    all_conll.append(data)
    df['sentence_id']+=offset
    offset+=df.shape[0]
    dfs.append(df)
  return '\n\n'.join(all_conll), pd.concat(dfs)
    
def word2features(sent, i):
    word = sent[i]
    postag = word.pos_
    word=str(word)

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1]
        postag1 = word1.pos_
        word1=str(word1)
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        postag1 = word1.pos_
        word1=str(word1)
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]




def compute_metrics_and_averages(y_true, y_pred):
    def extract_entities(sentence_tags, row_id):
        entities = []
        current_entity = None
        for i, tag in enumerate(sentence_tags):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'type': tag[2:], 'start': i, 'end': i, 'row_id': row_id}
            elif tag.startswith('I-') and current_entity and current_entity['type'] == tag[2:]:
                current_entity['end'] = i
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)
        return entities
 
    # Initialize containers
    metrics = {}
    FP_ids = {}
    FN_ids = {}
 
    for row_id, (true_tags, pred_tags) in enumerate(zip(y_true, y_pred)):
        true_entities = extract_entities(true_tags, row_id)
        pred_entities = extract_entities(pred_tags, row_id)
 
        for entity in true_entities + pred_entities:
            entity_type = entity['type']
            if entity_type not in metrics:
                metrics[entity_type] = {'TP': 0, 'FP': 0, 'FN': 0}
                FP_ids[entity_type] = []
                FN_ids[entity_type] = []
 
        for pred_entity in pred_entities:
            matched = False
            for true_entity in true_entities:
                if pred_entity['type'] == true_entity['type'] and not (pred_entity['end'] < true_entity['start'] or pred_entity['start'] > true_entity['end']):
                    metrics[pred_entity['type']]['TP'] += 1
                    matched = True
                    true_entities.remove(true_entity)
                    break
            if not matched:
                metrics[pred_entity['type']]['FP'] += 1
                FP_ids[pred_entity['type']].append(pred_entity['row_id'])
 
        for true_entity in true_entities:
            metrics[true_entity['type']]['FN'] += 1
            FN_ids[true_entity['type']].append(true_entity['row_id'])
 
    # Calculate micro and macro averages
    total_TP = sum(metrics[etype]['TP'] for etype in metrics)
    total_FP = sum(metrics[etype]['FP'] for etype in metrics)
    total_FN = sum(metrics[etype]['FN'] for etype in metrics)
 
    micro_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0
 
    precisions = [metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FP']) if metrics[etype]['TP'] + metrics[etype]['FP'] > 0 else 0 for etype in metrics]
    recalls = [metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FN']) if metrics[etype]['TP'] + metrics[etype]['FN'] > 0 else 0 for etype in metrics]
    macro_precision = sum(precisions) / len(metrics) if metrics else 0
    macro_recall = sum(recalls) / len(metrics) if metrics else 0
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if macro_precision + macro_recall > 0 else 0
 
    # Prepare DataFrame
    data = {
        'Entity Type': list(metrics.keys()) + ['Micro Average', 'Macro Average'],
        'Precision': [metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FP']) if metrics[etype]['TP'] + metrics[etype]['FP'] > 0 else 0 for etype in metrics] + [micro_precision, macro_precision],
        'Recall': [metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FN']) if metrics[etype]['TP'] + metrics[etype]['FN'] > 0 else 0 for etype in metrics] + [micro_recall, macro_recall],
        'F1': [2 * (metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FP']) * metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FN'])) / ((metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FP'])) + (metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FN']))) if (metrics[etype]['TP'] + metrics[etype]['FP'])>0 and (metrics[etype]['TP'] + metrics[etype]['FN'])>0 and(metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FP'])) + (metrics[etype]['TP'] / (metrics[etype]['TP'] + metrics[etype]['FN'])) > 0 else 0 for etype in metrics] + [micro_f1, macro_f1]
    }
 
    results_df = pd.DataFrame(data)
    return results_df, FP_ids, FN_ids, micro_precision, micro_recall, micro_f1

class CRFModel(object):
    """
    Wrap sklearn-crfsuit model with common functions used for ALSampler
    
    """
    def __init__(self, anno_types=[], topNUncertainToken=2):
        self.reset_model()
        self.anno_types=anno_types
        self.topNUncertainToken=topNUncertainToken

    def reset_model(self):
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)

    def __df2features(self, df:pd.DataFrame):
        # need to keep the sentence_id order-- groupby will sort
        original_idx=[]
        X=[]
        y=[]
        for idx, sdf in df.groupby('sentence_id'):
            original_idx.append(sdf.index[0])
            X.append(sent2features(list(sdf['token'])))
            y.append(list(sdf['label']))
        sorted_X=sorted(zip(original_idx, X), key=lambda x: x[0])
        X=[x for i,x in sorted_X]
        sorted_y=sorted(zip(original_idx, y), key=lambda x: x[0])
        y=[y for i,y in sorted_y]
        return X,y

    def df2features(self, df:pd.DataFrame):
        # need to keep the sentence_id order-- groupby will sort
        original_idx=[]
        X=[]
        y=[]
        for idx, sdf in df.groupby('sentence_id'):
            original_idx.append(sdf.index[0])
            X.append(sent2features(list(sdf['token'])))
            y.append(list(sdf['label']))
        sorted_X=sorted(zip(original_idx, X), key=lambda x: x[0])
        X=[x for i,x in sorted_X]
        sorted_y=sorted(zip(original_idx, y), key=lambda x: x[0])
        y=[y for i,y in sorted_y]
        return X,y
        
    def fit(self, docs: List[Doc]):
        _, train_df=convert_docs(docs, anno_types=self.anno_types)
        X_train, y_train=self.__df2features(train_df)
        logger.debug('Reset and train CRF model...')
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)
        self.crf.fit(X_train, y_train)
        logger.debug('Training complete.')

    def transform(self, docs:List[Doc]):
        '''
        Extract origianl NER annotation
        Also using model to predict the labels on the given doc list
        '''
        _, test_df=convert_docs(docs, anno_types=self.anno_types)
        X_test, y_test=self.__df2features(test_df) #extract the origianl label
        y_pred = self.crf.predict(X_test) #predict the NER label
        return y_test, y_pred 

    def eval(self, docs:List[Doc]):
        logger.debug('Predicting eval docs...')
        y_test, y_pred =self.transform(docs)
        logger.debug('Calculate scores...')
        results_df, FP_ids, FN_ids, micro_precision, micro_recall, micro_f1=compute_metrics_and_averages(y_test, y_pred)
        logger.debug('Complete.')
        return results_df, FP_ids, FN_ids, micro_precision, micro_recall, micro_f1

    def eval_scores(self, docs:List[Doc]):
        results_df, FP_ids, FN_ids, micro_precision, micro_recall, micro_f1=self.eval(docs)
        return {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}

    def bootstrap_eval(self, docs:List[Doc], times=200):
        logger.debug('Predicting eval docs...')
        y_test, y_pred =self.transform(docs)
        sample_size=len(y_test)
        bootstrap_scores={'precision':[], 'recall':[],'f1':[]}
        logger.debug(f'Calculate scores from bootstrapping {times} times')
        for _ in range(times):
            random.seed(_)
            sample_indices=[random.randint(0,sample_size-1) for _ in range(sample_size)]
            sy_test=[y_test[i] for i in sample_indices]
            sy_pred=[y_pred[i] for i in sample_indices]
            results_df, FP_ids, FN_ids, micro_precision, micro_recall, micro_f1=compute_metrics_and_averages(sy_test, sy_pred)
            bootstrap_scores['precision'].append(micro_precision)
            bootstrap_scores['recall'].append(micro_recall)
            bootstrap_scores['f1'].append(micro_f1)
        logger.debug('complete')
        return bootstrap_scores
            

    def compute_mean_ci(self, scores):
        ave=np.mean(scores)
        ci=np.percentile(scores, [2.5, 97.5])
        return ave, ci
        
    def estimate_certainties(self, docs:List[Doc]): #Document level certainties
        _, df=convert_docs(docs, anno_types=self.anno_types)
        X, y=self.__df2features(df)
        yp=self.crf.predict_marginals(X)
        sdf=df.drop_duplicates('sentence_id')[['doc_name', 'sentence_id']].reset_index(drop=True)
        sentence_certainties=self.calculate_certainties(yp)
        sdf['certainty']=sentence_certainties
        # now we need to estimate the certainty per doc, because we need to decide which document to sample instead of sentence.
        result=sdf.groupby('doc_name')['certainty'].min().reset_index()
        d_certainties={}
        for i, r in result.iterrows():
            d_certainties[r['doc_name']]=r['certainty']
        return [d_certainties[d._.doc_name] for d in docs]
                
    def calculate_certainties(self, yp):
        return [self.single_sent_certainty(s) for s in yp]
    
    def single_sent_certainty(self, sent_logits):
        certainty=[]
        topNUncertainToken=self.topNUncertainToken
        for w in sent_logits:
            top2=sorted(w.values())[-2:]
            certainty.append(top2[1]-top2[0])
        certainty=sorted(certainty)        
        if topNUncertainToken>len(certainty):
            topNUncertainToken=len(certainty)
        return sum(certainty[:topNUncertainToken])/topNUncertainToken
        
        
        