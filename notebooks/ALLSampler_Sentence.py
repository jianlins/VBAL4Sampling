from spacy.tokens import Doc
from abc import abstractmethod
from typing import List, Tuple, Union
import random
from loguru import logger
import numpy as np
import pandas as pd
import faiss
from medspacy_io.vectorizer import Vectorizer

def sent_count(df:pd.DataFrame):
    '''
    The input dataframe should countain 4 columns: sentence_id	doc_name	token	label
    Ouput: number of sentences, list of unique sentence ID
    '''
    sentID_list = df['sentence_id'].to_list()
    sentID_set = set(sentID_list)
    sentID_uniqList = list(sentID_set)
    return len(sentID_uniqList), sentID_uniqList

def rand_sample(data, num=10, seed=14):
    '''
    data: any data list, can be documents or sentences
    randomly sample one time
    '''
    if num>=len(data):
        return data, []
    random.seed(seed)  # initialize default seed=14 random-number generators
    sampled_indices=random.sample(range(len(data)), num); # randomly sample num value from Data
    logger.debug(f'sampled_indices: {len(sampled_indices)}')
    sampeld_sublist=[data[i] for i in sampled_indices] # form sampled subset
    logger.debug(f'sampeld_sublist: {len(sampeld_sublist)}')
    sampled_indices=set(sampled_indices) # unique indices of the sample
    remaining_sublist = [d for i,d in enumerate(data) if i not in sampled_indices] # remaining un-sampled data
    return sampeld_sublist, remaining_sublist

def rand_sample_dfSentence(df:pd.DataFrame, num_sent=10, seed_sent=14):
    '''
    Randomly select num_sentences from dataframe
    '''
    if num_sent>=len(df.index):
        df_remain = pd.DataFrame() #empty remaining dataframe
        return df, df_remain
    # get list of sentenceID and sample on unique sentence ID 
    num_df_sent, list_df_sentID = sent_count(df)
    sampeld_sublist, remaining_sublist = rand_sample(list_df_sentID, num=num_sent, seed=seed_sent)
    # sampled sentence df by sentence not by rowID # REMARK: Make sure the sentence_id is int 32
    sampled_df = df[df['sentence_id'].isin(sampeld_sublist)]
    remained_df = df[df['sentence_id'].isin(remaining_sublist)]
    return sampled_df, remained_df



    
class SamplingSimulator():
    def __init__(self, 
                 total_sents:pd.DataFrame=None, 
                 total_round:int=10, 
                 modelWrapper:object=None, 
                 eval_sents:pd.DataFrame=None, 
                 init_seed:int=14, 
                 sample_all_on_last_round:bool=True,
                 **kwargs):
        self.total_sents = total_sents
        self.total_round = total_round #1/10 of the total number of sentence
        self.modelWrapper = modelWrapper
        self.eval_sents = eval_sents
        self.init_seed = init_seed
        self.sample_all_on_last_round=sample_all_on_last_round
        # calculate number of unique sentences
        num_df_sent, list_df_UniqSentID = sent_count(total_sents)
        self.num_per_round = int(1.0*len(list_df_UniqSentID)/total_round) 
        logger.debug(f'num per found unique sent: {self.num_per_round}')
        #2589 #1/20 sentences 10 rounds 
        #int(1.0*len(list_df_UniqSentID)/total_round)
        for k,v in kwargs.items():
            setattr(self, k, v)


    def keep_sample(self, first_round:bool=True):
        if first_round:
            logger.debug('The first round sampling will be random')
            self.sampled, self.remaining = self.sample_next_round(pd.DataFrame(), self.total_sents, True) #from entire doc dataset, sample 1st round                
        else:
            logger.debug('Sample according to certainties')
            self.sampled, self.remaining = self.sample_next_round(self.sampled, self.remaining, False) #from remaining doc subdataset, sample docs
        num_df_sent_sampled, list_df_UniqSentID_sampled = sent_count(self.sampled)
        num_df_sent_remained, list_df_UniqSentID_remained = sent_count(self.remaining)
        logger.info(f'current sampled sentences: {len(list_df_UniqSentID_sampled)}, remaining sentences: {len(list_df_UniqSentID_remained)}')
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
                self.sampled=self.total_sents
                self.remaining=pd.DataFrame() #last round: empty remaining
                logger.info("It's the last round, now add all docs to sampled.")
            # update model with new sampled before evaluate it.
            self.modelWrapper.reset_model()
            self.modelWrapper.fit(self.sampled) #each round train CRF model and test on testing set
            round_scores=self.modelWrapper.bootstrap_eval_DFsent(self.eval_sents, boostrap_times) #the scores for each sampling round
            mean_scores={k:np.mean(v) for k,v in round_scores.items()} #mean_score of all the rounds
            logger.debug(f'{mean_scores}')
            scores.append(round_scores)
            if len(self.remaining)==0:
                break
        return scores
            
    @abstractmethod
    def sample_next_round(self, sampled, remaining, randomly=True)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """place holder method to inidate the interface needed
        The following interfaces initialized this methods:
        RandomSamplingSimulator
        """
        pass


class RandomSamplingSimulator(SamplingSimulator):
    def sample_next_round(self, sampled, remaining, randomly=True):
        new_sampled, new_remaining=rand_sample_dfSentence(remaining, self.num_per_round, self.init_seed)
        updated_sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
        #sampled=sampled+new_sampled
        return updated_sampled, new_remaining
        
    
class ModelSamplingSimulator(SamplingSimulator):

    def sample_next_round(self, sampled, remaining, randomly=True):               
        if randomly: #first round randomly
            new_sampled, new_remaining=rand_sample_dfSentence(remaining, self.num_per_round, self.init_seed)
            updated_sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
            #sampled=sampled+new_sampled
            return updated_sampled, new_remaining

        #count the sentences when remaining or sampled are not empty
        num_df_sent_remained, list_df_UniqSentID_remained = sent_count(remaining)
        num_df_sent_sampled, list_df_UniqSentID_sampled = sent_count(sampled)

        #last round
        if self.num_per_round> num_df_sent_remained: 
            updated_sampled = pd.concat([sampled, remaining], ignore_index=True, axis = 0)
            return updated_sampled, pd.DataFrame()
        logger.debug(f'Train model wrapper on sampled {len(list_df_UniqSentID_sampled)} sentences samples')        
        logger.debug(f'Use trained model to estimate the remaining data certainty.')

        #other rounds: count sentences in remaining DF and sampled DF
        # now remaining has 3 columns: doc_name, sentence_id, certainty; each certainty is for one sentence
        certainties=self.modelWrapper.estimate_certainties(remaining) ## now remaining has 3 columns: doc_name, sentence_id, certainty
        certainties_descend = certainties.sort_values(by='certainty',ascending=True) #pick the most uncertain
        #sorted_idx=np.argsort(certainties['certainty'].to_list()) ## index of the certainties from high to low
        certaintyList = certainties['certainty'].to_list()
        
        logger.debug(f'remain {len(remaining.index)} rows, sort indx on certainty for {len(certaintyList)} sentences')
        if self.num_per_round>len(certaintyList):
            self.num_per_round=len(certaintyList)  

        #now the sampled data are DF 
        new_sampled_sentID = certainties_descend['sentence_id'].to_list()[:self.num_per_round]
        new_remaining_sentID = certainties_descend['sentence_id'].to_list()[self.num_per_round:]
        new_sampled = remaining[remaining['sentence_id'].isin(new_sampled_sentID)]
        new_remaining = remaining[remaining['sentence_id'].isin(new_remaining_sentID)]

        # update the sampled dataframe
        sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
        
        #new_sampled=[remaining[i] for i in sorted_idx[:self.num_per_round]] ## ERROR! Now the remaining and sampled are dataframe
        #new_remaining=[remaining[i] for i in sorted_idx[self.num_per_round:]]
        #sampled=sampled+new_sampled
        logger.debug('Update model with new sampled data')
        return sampled, new_remaining 




def convert_docs_medspacyIOvec(docs:List[Doc]):
    '''
    return a dataframe: sentence	concept	y	doc_name
    '''
    sdf_labels=Vectorizer.docs_to_sents_df(docs, track_doc_name=True).rename(columns={"X":"sentence"})
    
    #DONOT add sentence id but merge this with embedding to have consistence sid
    #uniq_sentSet = set(sdf_labels['sentence'].to_list())
    #uniq_sentList = list(uniq_sentSet)
    #sentIndexDic = {}
    #for i in range(len(uniq_sentList)):
    #    sentIndexDic[uniq_sentList[i]] = i
    #sendIDlist = []
    #for s in sdf_labels['sentence'].to_list():
    #    sendIDlist.append(sentIndexDic[s])
    #sdf_labels['sentence_id']=sendIDlist
    return sdf_labels

    

class VBSamplingSimulator(SamplingSimulator):
    def __init__(self, 
                 total_sents:pd.DataFrame=None, 
                 total_round:int=10, 
                 modelWrapper:object=None, 
                 eval_sents:pd.DataFrame=None, 
                 init_seed: int =14, 
                 sample_all_on_last_round: bool = True,
                 faiss_index_path:str=None,
                 min_sent_length:int=10, 
                 max_retrieve:Union[int,None]=None,
                 embedding_df:pd.DataFrame=None, 
                 sdf_labels:pd.DataFrame=None,#training docs convert to: sentence	concept	y	doc_name; must merged with embedding to have sentence id
                 min_dist_diff:bool=False,
                ):
        super().__init__(total_sents=total_sents, 
                         total_round=total_round, 
                         modelWrapper=modelWrapper, 
                         eval_sents=eval_sents, 
                         init_seed=init_seed, 
                         sample_all_on_last_round=sample_all_on_last_round,
                         faiss_index_path=faiss_index_path,
                         min_sent_length=min_sent_length,
                         max_retrieve=max_retrieve, 
                         embedding_df=embedding_df, #embedding vector DF: sentence_id	sentence	embedding
                         sdf_labels = sdf_labels, #format: sentence	concept	y	doc_name	sentence_id	embedding
                         min_dist_diff=min_dist_diff)
    
        logger.debug('Loading index...')
        self.index=faiss.read_index(faiss_index_path) #vector search
        logger.debug('done')
        
        # store all the labels with sentences to be used for computing centroids later.
        # This is from medspacy_io/vectorizer.docs_to_sents_df will return 4 columns
        # column X the text of context sentences; column concepts the text of labeled concepts, y is the label; doc_name is docment id
        # This step is running before sampler  convert_docs_medspacyIOvec
        # the sid column is int64 which is not int type sdf_labels_intSid = vb_simulator.sdf_labels.astype({'sid':'int'})
        sdf_labels_intSid = sdf_labels.astype({'sentence_id':'int'}) #({'sid':'int'})
        self.sdf_labels = sdf_labels_intSid        
      
        # total number of sentences are count from new embedding for unique sentences
        self.total_sents_num=self.embedding_df.shape[0]
        
        # now we only need a sentence to embedding dictionary, so no need duplicates
        self.embedding_df[['sentence', 'embedding']].drop_duplicates(subset='sentence',keep='first', inplace=True)

        #initial centroid
        self.centroid={}
        self.num_centroid=0
        if max_retrieve is None:
            self.max_retrieve=self.total_sents_num-self.num_per_round+1

            

    def fit(self, sampled_sents):
        '''
        Input: sampled dataframe: sentence	concept	y	doc_name  sentence_id
        return: cetroid dictionary: label: centroid vector
                centroid number: num_centroid #number of concepts
        convert_docs_medspacyIOvec(docs:List[Doc]):
    return a dataframe: sentence	concept	y	doc_name  sentence_id
        '''
        sampled_sendIDlist = sampled_sents['sentence_id'] 
        sampled_sdf = self.sdf_labels[self.sdf_labels['sentence_id'].isin(list(set(sampled_sendIDlist)) )]
        self.centroid={t:self.compute_mean(v) for t, v in sampled_sdf.groupby('y')}#group by label: t is grouped y label; v is df under this group
        self.num_centroid=len(self.centroid)
        logger.debug(f'{self.num_centroid} centroids detected from the given sampled_docs')
        

    def compute_mean(self, sdf_single_label:pd.DataFrame):
        single_label_embeddings=sdf_single_label.merge(self.embedding_df, how='inner', on='sentence')        
        return np.mean(np.array(single_label_embeddings.embedding.tolist()), axis=0)

    def sort_dist_diff(self, distances:np.ndarray)->np.ndarray:
        if self.min_dist_diff:
            logger.debug('Compute min difference between distances to any two centroids.')
            rows, cols=distances.shape #AFTER FAISS INDEXING distance shape: (51798, 4) four centroid for all sentences; rows=51798, columns=4
            differences=np.abs(distances[:, None, :] -distances[:,:,None]) #element wise absolute value
            logger.debug(f'min_dist diferences shape: {differences.shape}')
            triu_idx=np.triu_indices(cols, k=1)
            non_diag_diffs=differences[:, triu_idx[0], triu_idx[1]]            
            min_differences=np.amin(non_diag_diffs, axis=1)
            
            mask= ~np.isnan(min_differences) #sent_id is not consecutive, the distances is of size max_sent_id, so there will be a lot of nan when sent_id does not appear
            stacked_array=np.vstack((min_differences[mask], np.arange(len(distances))[mask])).T #stack row wise then transpose len_valid_dist x 2 shape
            sorted_data = stacked_array[-stacked_array[:,0].argsort()] #sort descending according to distances; pick max distance next round
            sorted_sids=sorted_data[:,1].astype(int)
        else:
            logger.debug('Compute max distance differences to centroids.')
            max_values=np.amax(distances, axis=1) #max of each row
            min_values=np.amin(distances, axis=1)        
            logger.debug("DISTANCE max_values each row", max_values, "size:", len(max_values))
            logger.debug("DISTANCE min_values each row", min_values, "size:", len(min_values))
            max_differences=max_values-min_values
            logger.debug("DISTANCE max_diff each row", max_differences, "size:", len(max_differences))
    
            mask= ~np.isnan(max_differences) #test element wise for NAN return boolean vec
            logger.debug("after masking max_diff size", len(max_differences[mask]) )
            stacked_array=np.vstack((max_differences[mask], np.arange(len(distances))[mask])).T #stack row wise then transpose len_valid_dist x 2 shape
            sorted_data = stacked_array[-stacked_array[:,0].argsort()] #sort descending according to distances; pick max distance next round
            sorted_sids=sorted_data[:,1].astype(int)
        return sorted_sids
            
    def sample_next_round(self, sampled, remaining, randomly=True):
        if randomly: # first round randomly
            new_sampled, new_remaining=rand_sample_dfSentence(remaining, self.num_per_round, self.init_seed)
            updated_sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
            #sampled=sampled+new_sampled
            return updated_sampled, new_remaining

        #count the sentences when remaining or sampled are not empty
        num_df_sent_remained, list_df_UniqSentID_remained = sent_count(remaining)
        num_df_sent_sampled, list_df_UniqSentID_sampled = sent_count(sampled)

        #last round
        if self.num_per_round> num_df_sent_remained:
            updated_sampled = pd.concat([sampled, remaining], ignore_index=True, axis = 0)
            logger.info(f'Not enough documents left to sample {self.num_per_round} document. Add them all {len(remaining)} in this round.')
            return updated_sampled, pd.DataFrame()
                        
        logger.debug('Calculating centroids...')
        self.fit(sampled)
        
        logger.debug('Searching from the vector index...')

        
        # use numpy ops to speed up, set default value to np.nan (distance should be >=0), so that we can tell which cell has not been updated
        # REMARK: sentence ID is not labeled consecutively. Distance should be initialized according to the max value of sentence_id, otherwise distance[sid,ci] will exceed the index boundary
        sentID_max = max(self.total_sents['sentence_id'].to_list())
        distances=np.full((sentID_max+1, self.num_centroid), np.nan)
        #distances=np.full((self.total_sents_num, self.num_centroid), np.nan) #fill matrix size total number of sentences X num centroid
        
        max_retrieve=self.max_retrieve
        if self.total_sents_num< max_retrieve:
            max_retrieve=self.total_sents_num
        #estimated_needed_sents=self.num_per_round*300
        # if estimated_needed_sents<max_retrieve:
        #     max_retrieve=estimated_needed_sents
        
        logger.debug(f'AFTER FAISS INDEXING distance shape: {distances.shape}, max to retrieve {max_retrieve} sentences')
        # list distances to all centroid for each sid, then find the most uncertain ones---the difference between distances are small to at least two centroid
        for ci, (t,v) in enumerate(self.centroid.items()): 
        # this can be optimzed to limit to a smaller subset, when dealing with large corpus, no need to sort them all
            logger.debug(f'search for centroid: {t}') # t is label
            D, I=self.index.search(v.reshape(1, len(v)), max_retrieve) #faiss index search for centroid vector v
            logger.debug(f'FAISS index D shape {D.shape}, I shape: {I.shape}')
            for d, sid in zip(D[0], I[0]): # fill in all the distances value, the sampled ones will be excluded later
                #logger.debug(f'current round: Distance: {d}, sentID: {sid}') ### REMARK SENTENCE ID is not consecutive, not row index
                distances[sid, ci]=d
                
                    
        # isolate the sorting logic for easier debugging
        logger.debug(f'AFTER FAISS INDEXING distance shape: {distances.shape}')
        sorted_sids=self.sort_dist_diff(distances)
        
        logger.debug(f'The Sorted sentence IDs according to the distances: the sorted_sids length is: {len(sorted_sids)}')
        #print("the sorted sids:", sorted_sids)

        # REMARK: THE sorted_sids includes all sentences_id, the sampled one should be exclude, otherwise, the sampled ones are always the ones closest to the the centroid! THERE IS THE NEW UPDATES
        sorted_sid_excludeSampled = list( set(sorted_sids)-set(sampled['sentence_id'].to_list()) ) # the sorted order is still maintained
        logger.debug(f'AFTER EXCLUDE SAMPLED SENT ID: the sorted_sid_excludeSampled length is: {len(sorted_sid_excludeSampled)}')
        #print("the sorted sids removing the sampled:", sorted_sid_excludeSampled) #Remark: this is different from remaining because sorted_ids exclude the NAN dist

        # from the sorted distances, find a num_per_round list of sentence ID that has the largest distance
        new_sampled_sentID = set()
        for sid in sorted_sid_excludeSampled:
            new_sampled_sentID.add(sid)
            if len(new_sampled_sentID)>=self.num_per_round:
                logger.debug('NEW SAMPLED UNIQUE SENT REACH NUM_PER_ROUND---:')
                logger.debug(f'num per found {self.num_per_round}')
                logger.debug(f'new_sampled_sentID: {len(new_sampled_sentID)}')
                logger.debug('--------------------------')
                break;
        #print("new_sampled_sentID is Unique Sent in sortedID:", new_sampled_sentID)
        new_remaining_sentID = set(remaining['sentence_id'].to_list())-new_sampled_sentID
        logger.debug(f'new sampled unique sent num: {len(new_sampled_sentID)}')
        logger.debug(f'new remaining unique sent num: {len(new_remaining_sentID)}')

        #now the sampled data are DF
        new_sampled = remaining[remaining['sentence_id'].isin(list(new_sampled_sentID))]
        new_remaining = remaining[remaining['sentence_id'].isin(list(new_remaining_sentID))]
        logger.debug(f'BEFORE update model with old sampled data {len(sampled)}, old remaining data {len(remaining)} ')
        #print("sampled sentID before updating:", list( set(sampled['sentence_id'].to_list()) ) )
        #print("new sampled current round:", new_sampled_sentID )
        logger.debug(f'new_sampled {len(new_sampled)}, new remaining data {len(new_remaining)} ')
        # update the sampled dataframe
        sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
        logger.debug(f'AFTER update model with new sampled data {len(sampled)}, new remaining data {len(new_remaining)} ')
        #print("sampled sentID after updating:", list( set(sampled['sentence_id'].to_list()) ) )
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
            
               



        
        
        

