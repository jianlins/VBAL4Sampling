from spacy.tokens import Doc
from abc import abstractmethod
from typing import List, Tuple, Union
import random
from loguru import logger
import numpy as np
import pandas as pd
import faiss
from medspacy_io.vectorizer import Vectorizer

def rand_sample(data, num=10, seed=14):
    '''
    data: any data list, can be documents or sentences
    randomly sample one time
    '''
    if num>=len(data):
        return data, []
    random.seed(seed)  # initialize default seed=14 random-number generators
    sampled_indices=random.sample(range(len(data)), num); # randomly sample num value from Data
    sampeld_sublist=[data[i] for i in sampled_indices] # form sampled subset
    sampled_indices=set(sampled_indices) # unique indices of the sample
    remaining_sublist = [d for i,d in enumerate(data) if i not in sampled_indices] # remaining un-sampled data
    return sampeld_sublist, remaining_sublist
    
class SamplingSimulator():
    def __init__(self, total_docs:List[Doc], total_round:int=10, modelWrapper:object=None, eval_docs:List[Doc]=[], init_seed=14, sample_all_on_last_round:bool=True, **kwargs):
        self.total_docs=total_docs
        self.total_round=total_round
        self.modelWrapper=modelWrapper
        self.eval_docs=eval_docs
        self.init_seed=4
        self.num_per_round=int(1.0*len(total_docs)/total_round)
        self.sample_all_on_last_round=sample_all_on_last_round
        for k,v in kwargs.items():
            setattr(self, k, v)


    def keep_sample(self, first_round:bool=True):
        if first_round:
            logger.debug('The first round sampling will be random')
            self.sampled, self.remaining=self.sample_next_round([], self.total_docs, True) #from entire doc dataset, sample 1st round                
        else:
            self.sampled, self.remaining=self.sample_next_round(self.sampled, self.remaining, False) #from remaining doc subdataset, sample docs
        logger.info(f'current sampled: {len(self.sampled)}, remaining: {len(self.remaining)}')
        return self.sampled, self.remaining
        
    def simulate_rounds(self, boostrap_times=200):
        '''
        train model on sampled dataset
        boostrap_times is for bootstrapping evaluation
        '''
        scores=[]
        for i, r in enumerate(range(self.total_round)):
            logger.info(f'simulate round {i}.')            
            self.keep_sample(i==0) 
            if self.sample_all_on_last_round and i==self.total_round-1:
                self.sampled=self.total_docs
                self.remaining=[]
                logger.info("It's the last round, now add all docs to sampled.")
            # update model with new sampled before evaluate it.
            self.modelWrapper.reset_model()
            self.modelWrapper.fit(self.sampled) #each round train CRF model and test on testing set
            round_scores=self.modelWrapper.bootstrap_eval(self.eval_docs, boostrap_times) #the scores for each sampling round
            mean_scores={k:np.mean(v) for k,v in round_scores.items()}
            logger.debug(f'{mean_scores}')
            scores.append(round_scores)
            if len(self.remaining)==0:
                break
        return scores
            
    @abstractmethod
    def sample_next_round(self, sampled, remaining, randomly=True)-> Tuple[List[Doc], List[Doc]]:
        """place holder method to inidate the interface needed
        The following interfaces initialized this methods:
        RandomSamplingSimulator
        """
        pass


class RandomSamplingSimulator(SamplingSimulator):
    def sample_next_round(self, sampled, remaining, randomly=True):
        new_sampled, new_remaining=rand_sample(remaining, self.num_per_round, self.init_seed)
        sampled=sampled+new_sampled
        return sampled, new_remaining
        
    
class ModelSamplingSimulator(SamplingSimulator):

    def sample_next_round(self, sampled, remaining, randomly=True):
        if randomly:
            new_sampled, new_remaining=rand_sample(remaining, self.num_per_round, self.init_seed)
            sampled=sampled+new_sampled
            return sampled, new_remaining
        if self.num_per_round> len(remaining):
            return sampled+remaining, []
        logger.debug(f'Train model wrapper on sampled {len(sampled)} samples')        
        logger.debug(f'Use trained model to estimate the remaining data certainty.')
        certainties=self.modelWrapper.estimate_certainties(remaining)
        sorted_idx=np.argsort(certainties)
        logger.debug(f'remain {len(certainties)} docs, sort indx on certainty for {len(certainties)}')
        if self.num_per_round>len(certainties):
            self.num_per_round=len(certainties)  
        
        new_sampled=[remaining[i] for i in sorted_idx[:self.num_per_round]]
        new_remaining=[remaining[i] for i in sorted_idx[self.num_per_round:]]
        sampled=sampled+new_sampled
        logger.debug('Update model with new sampled data')
        return sampled, new_remaining 



class VBSamplingSimulator(SamplingSimulator):
    def __init__(self, total_docs:List[Doc], total_round:int=10, modelWrapper:object=None, eval_docs:List[Doc]=[], init_seed=14, sample_all_on_last_round:bool=True, faiss_index_path:str=None,
                 min_sent_length:int=10, max_retrieve:Union[int,None]=None, embedding_df:pd.DataFrame=None, min_dist_diff:bool=False):
        """embedding_df to keep the generated embeddings, this will slower for large corpus, but fast when it's small.
            min_dist_diff: if true, prioritize the sentences that have distances to two centroids have smaller difference
                            if false, then prioritize the sentences that have a smaller difference of the max differences to all centroids (max distance- min distance).            
        """
        super().__init__(total_docs=total_docs, total_round=total_round, modelWrapper=modelWrapper, eval_docs=eval_docs, init_seed=init_seed, sample_all_on_last_round=sample_all_on_last_round,
                         faiss_index_path=faiss_index_path, min_sent_length=min_sent_length, max_retrieve=max_retrieve, embedding_df=embedding_df, min_dist_diff=min_dist_diff)
        logger.debug('Loading index...')
        self.index=faiss.read_index(faiss_index_path)
        logger.debug('done')
        # store all the labels with sentences to be used for computing centroids later.
        self.sdf_labels=Vectorizer.docs_to_sents_df(total_docs, track_doc_name=True).rename(columns={"X":"sentence"})
        self.sid2doc_name={r['sid']:r['doc_name'] for i,r in self.embedding_df.iterrows()}
        # expecting to have 4 columns: sid, sentence, doc_name, embedding with directly constructed from doc.sents without taking entitiy labels
        self.total_sents=self.embedding_df.shape[0]
        # now we only need a sentence to embedding dictionary, so no need duplicates
        self.embedding_df[['sentence', 'embedding']].drop_duplicates(subset='sentence',keep='first', inplace=True)
        self.centroid={}
        self.num_centroid=0
        if max_retrieve is None:
            self.max_retrieve=self.total_sents-self.num_per_round+1
        

    # def convert2df(self, docs: List[Doc], min_length:int=10):
    #     data={'sentence':[], 'doc_name':[]}
    #     for d in docs:
    #         sents=[str(s) for s in d.sents if len(str(s).strip())>min_length]
    #         data['sentence']+=sents
    #         data['doc_name']+=[d._.doc_name]*len(sents)
    #     return pd.DataFrame(data).rename_axis('sid').reset_index()
            
        
    def fit(self, sampled_docs):
        # this fit is not training a model, but try to compute the centroid for each label
        doc_names={d._.doc_name for d in sampled_docs}
        sampled_sdf=self.sdf_labels[self.sdf_labels.doc_name.isin(doc_names)]
        self.centroid={t:self.compute_mean(v) for t, v in sampled_sdf.groupby('y')}
        self.num_centroid=len(self.centroid)
        logger.debug(f'{self.num_centroid} centroids detected from the given sampled_docs')

    def compute_mean(self, sdf_single_label:pd.DataFrame):
        single_label_embeddings=sdf_single_label.merge(self.embedding_df, how='inner', on='sentence')        
        return np.mean(np.array(single_label_embeddings.embedding.tolist()), axis=0)

    def sort_dist_diff(self, distances:np.ndarray)->np.ndarray:
        if self.min_dist_diff:
            logger.debug('Compute min difference between distances to any two centroids.')
            rows, cols=distances.shape
            differences=np.abs(distances[:, None, :] -distances[:,:,None])
            triu_idx=np.triu_indices(cols, k=1)
            non_diag_diffs=differences[:, triu_idx[0], triu_idx[1]]            
            min_differences=np.amin(non_diag_diffs, axis=1)
            mask= ~np.isnan(min_differences)
            sorted_data=np.vstack((min_differences[mask], np.arange(len(distances))[mask])).T
            sorted_sids=sorted_data[:,1].astype(int)
        else:
            logger.debug('Compute max distance differences to centroids.')
            max_values=np.amax(distances, axis=1)
            min_values=np.amin(distances, axis=1)        
            
            max_differences=max_values-min_values        
    
            mask= ~np.isnan(max_differences)
            sorted_data=np.vstack((max_differences[mask], np.arange(len(distances))[mask])).T
            sorted_sids=sorted_data[:,1].astype(int)
        return sorted_sids
            
    def sample_next_round(self, sampled:List, remaining:List, randomly=True):
        if randomly:
            new_sampled, new_remaining=rand_sample(remaining, self.num_per_round, self.init_seed)
            sampled=sampled+new_sampled
            return sampled, new_remaining

        if len(remaining)<self.num_per_round:
            logger.info(f'Not enough documents left to sample {self.num_per_round} document. Add them all {len(remaining)} in this round.')
            return sampled+remaining, []
            
        logger.debug('Calculating centroids...')
        self.fit(sampled)
        
        logger.debug('Searching from the vector index...')
        remain_doc_names=set(d._.doc_name for d in self.remaining)
        # use numpy ops to speed up, set default value to np.nan (distance should be >=0), so that we can tell which cell has not been updated
        distances=np.full((self.total_sents, self.num_centroid), np.nan)
        max_retrieve=self.max_retrieve
        if self.total_sents< max_retrieve:
            max_retrieve=self.total_sents
        estimated_needed_sents=self.num_per_round*300
        # if estimated_needed_sents<max_retrieve:
        #     max_retrieve=estimated_needed_sents
        logger.info(f'distance shape: {distances.shape}, max to retrieve {max_retrieve} sentences')
        # list distances to all centroid for each sid, then find the most uncertain ones---the difference between distances are small to at least two centroid
        for ci, (t,v) in enumerate(self.centroid.items()): 
        # this can be optimzed to limit to a smaller subset, when dealing with large corpus, no need to sort them all
            logger.debug(f'search for centroid: {t}')
            D, I=self.index.search(v.reshape(1, len(v)), max_retrieve)
            for d, sid in zip(D[0], I[0]):                
                if self.sid2doc_name[sid] in remain_doc_names:
                    distances[sid, ci]=d
                    
        # isolate the sorting logic for easier debugging
        sorted_sids=self.sort_dist_diff(distances)
        
        logger.debug('Locate the docs of these sentences')
        
        new_sampled=set()
        for sid in sorted_sids:
            new_sampled.add(self.sid2doc_name[sid])
            if len(new_sampled)>=self.num_per_round:
                break;
        new_sampled_docs=[]
        new_remaining=[]
        for d in remaining:
            if d._.doc_name in new_sampled:
                new_sampled_docs.append(d)
            else:
                new_remaining.append(d)
        sampled+=new_sampled_docs        
        return sampled, new_remaining
        
        
def compute_mean_ci(scores):
    ave=np.mean(scores)
    ci=np.percentile(scores, [2.5, 97.5])
    return ave, ci
def summarize(scores):
    summary={'precision': [], 'pl':[], 'pu': [], 'recall': [], 'rl':[], 'ru': [], 'f1':[], 'fl':[], 'fu': []}
    for s in scores:    
        for k,v in s.items():
            ave, (l, u)=compute_mean_ci(v)
            summary[k].append(ave)
            summary[k[0]+'l'].append(l)
            summary[k[0]+'u'].append(u)
    return pd.DataFrame(summary)            
            
        
        
        
        
        
        

