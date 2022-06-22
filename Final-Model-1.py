#Imports
from pykeen.pipeline import pipeline

from pykeen.models import TransE
from pykeen.models import RESCAL
from pykeen.models import predict

from pykeen.datasets import FB15k237

from pykeen.models.predict import get_relation_prediction_df
from pykeen.models.predict import get_all_prediction_df
from pykeen.models.predict import predict_triples_df

from pykeen.datasets.analysis import get_relation_count_df
from pykeen.datasets.analysis import get_entity_count_df
#from pykeen.datasets.analysis import get_relation_functionality_df
#from pykeen.datasets.analysis import get_entity_relation_co_occurrence_df
from pykeen.datasets.analysis import get_relation_pattern_types_df

from pykeen.triples.triples_factory import tensor_to_df
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.triples.triples_factory import TriplesFactory

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from collections import Counter

import scipy.stats as stats

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from pykeen.triples import TriplesFactory
from pykeen.triples import CoreTriplesFactory
from pykeen.datasets.nations import NATIONS_TRAIN_PATH, NATIONS_TEST_PATH
import torch

from yellowbrick.cluster import KElbowVisualizer

from typing import Collection, Optional, Tuple
from math import ceil
from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
from random import sample


import torch

from pykeen.constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX
from pykeen.typing import Target

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids

import seaborn as sns

import itertools

import random

from random import randrange
import time

from statsmodels.stats.weightstats import ztest as ztest

#negative sampling class
class MyNegativeSampler(BasicNegativeSampler):
    r"""A basic negative sampler.

    This negative sampler that corrupts positive triples $(h,r,t) \in \mathcal{K}$ by replacing either $h$, $r$ or $t$
    based on the chosen corruption scheme. The corruption scheme can contain $h$, $r$ and $t$ or any subset of these.

    Steps:

    1. Randomly (uniformly) determine whether $h$, $r$ or $t$ shall be corrupted for a positive triple
       $(h,r,t) \in \mathcal{K}$.
    2. Randomly (uniformly) sample an entity $e \in \mathcal{E}$ or relation $r' \in \mathcal{R}$ for selection to
       corrupt the triple.

       - If $h$ was selected before, the corrupted triple is $(e,r,t)$
       - If $r$ was selected before, the corrupted triple is $(h,r',t)$
       - If $t$ was selected before, the corrupted triple is $(h,r,e)$
    3. If ``filtered`` is set to ``True``, all proposed corrupted triples that also exist as
       actual positive triples $(h,r,t) \in \mathcal{K}$ will be removed.
    """

    def __init__(
        self,
        *,
        triple: torch.Tensor,
        percentage: int,
        corruption_scheme: Optional[Collection[Target]] = None,
        **kwargs,
    ) -> None:
        """Initialize the basic negative sampler with the given entities.

        :param corruption_scheme:
            What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        
        super().__init__(**kwargs)
        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        # Set the indices
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        self.triple=triple
        self.percentage=percentage
        
        
        

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(ceil(total_num_negatives / len(self._corruption_indices)))

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)
            random_replacement_(
                batch=negative_batch,
                index=index,
                selection=slice(start, stop),
                size=stop - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
            )

        negative_factor = (self.percentage/100)
        positive_factor = (1-negative_factor)
        
        negative_batch_array = negative_batch.view(*batch_shape, self.num_negs_per_pos, 3).numpy()
        first_half = negative_batch_array[:int((len(negative_batch_array)*positive_factor))]
        second_half = negative_batch_array[int((len(negative_batch_array)/2)):]
        
        second_half_with_triple = []
        
        for i in range(int((len(negative_batch_array)*negative_factor))):
            second_half_with_triple.append([self.triple] )
        
        second_half_with_triple_array = np.array(second_half_with_triple)
        first_half_array = np.array(first_half)
        
        test_list1 = np.ndarray.tolist(first_half_array)
        test_list2 = np.ndarray.tolist(second_half_with_triple_array)
        
        negative_batch_with_triple_list = test_list1 + test_list2
        
        negative_batch_with_triple_array = np.array(negative_batch_with_triple_list)
        
        pt_tensor_from_list = torch.Tensor(negative_batch_with_triple_array)
        typecst = pt_tensor_from_list.type(torch.int64)
        
        #return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
        return  typecst

#function from Pykeen needed for the negative sampling class
def random_replacement_(batch: torch.LongTensor, index: int, selection: slice, size: int, max_index: int) -> None:
    """
    Replace a column of a batch of indices by random indices.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """
    # At least make sure to not replace the triples by the original value
    # To make sure we don't replace the {head, relation, tail} by the
    # original value we shift all values greater or equal than the original value by one up
    # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
    replacement = torch.randint(
        high=max_index - 1,
        size=(size,),
        device=batch.device,
    )
    replacement += (replacement >= batch[selection, index]).long()
    batch[selection, index] = replacement

#function to check for existing triples and replaces non-existing triples with existing
def check_and_replace_triple(reduced_list, triples_not_in_training_ids, num_relations):

    new_list = []
    stopper = True
    for idx, ele in enumerate(reduced_list):
        stopper = True
        while stopper:
            if ele in triples_not_in_training_ids:
                new_list.append(ele)
                stopper = False
            else:
                random_relation = randrange(num_relations)
                triple = [ele[0], random_relation, ele[2]]
                if triple in triples_not_in_training_ids:
                    new_list.append(triple)
                    stopper = False
                else:
                    stopper = True
                    
                
    return(new_list)

#function that rounds scores from a list of lists
def round_list(score_list):
    new_list = []
    for idx, ele in enumerate(score_list):
        ele = [round(num, 2) for num in ele]
        new_list.append(ele)
    return(new_list)

#track time
start = time.time()

#variables in the experiments
experiment_number = 'pos1'
dataset = FB15k237()
number_of_epochs = 13
loss = 'BCEWithLogitsLoss'
KGE_model = RESCAL
training_loop = 'sLCWA'
basic_negative_sampler = 'basic'
evaluator = 'RankBasedEvaluator'
model_kwargs = dict(predict_with_sigmoid=True, embedding_dim=50)
optimizer_kwargs = dict(lr=0.0001)
seed = int(3757357109) #seed 3757357109 -> has impact on the relations that are being tested, because it sets a seed for random.sample()
number_of_clusters = 3
number_of_relations_to_inspect = 11
positive_oversampling = float(1) #percentage of total triples
negative_oversampling = float(0.1) #percentage of negative batch to replace
metric = 'adjusted_arithmetic_mean_rank_index'
checkpoint_frequency_min = int(5)
checkpoint_S = '{}-S-{}epochs{}-seed:{}.pt'.format(experiment_number, number_of_epochs,loss,seed)
checkpoint_A_name = '{}-A-{}epochs{}-seed:{}.pt'.format(experiment_number, number_of_epochs,loss,seed)
checkpoint_B_name = '{}-B-{}epochs{}-seed:{}.pt'.format(experiment_number, number_of_epochs,loss,seed)
use_tqdm = False
device='cpu'

#Loading Situation S KGE
mapped_triples_training = dataset.training.mapped_triples
mapped_triples_testing = dataset.testing.mapped_triples

entity_to_id=dataset.training.entity_to_id
relation_to_id=dataset.training.relation_to_id

training_S = TriplesFactory(mapped_triples_training, entity_to_id, relation_to_id)
testing_S = TriplesFactory(mapped_triples_testing, entity_to_id, relation_to_id)

pipeline_result_s = pipeline(
    training=training_S,
    testing=testing_S,
    model= KGE_model,
    model_kwargs = model_kwargs,
    optimizer_kwargs = optimizer_kwargs,
    training_loop=training_loop,
    negative_sampler=basic_negative_sampler,
    evaluator=evaluator,
    loss=loss,
    training_kwargs=dict(
        num_epochs=number_of_epochs,
        checkpoint_name=checkpoint_S,
        checkpoint_frequency=checkpoint_frequency_min),
    random_seed = seed,
    device=device,
    use_tqdm=use_tqdm
    
)

evaluation_summary_s = pipeline_result_s.metric_results
losses_s = pipeline_result_s.losses

metric_df_s = evaluation_summary_s.to_df()
metric_df_s = metric_df_s.loc[(metric_df_s['Metric'] == metric)]

#some important variables
mapped_triples = dataset.training.mapped_triples

model_s = pipeline_result_s.model

triples_factory_s = pipeline_result_s.training

entity_labels = triples_factory_s.entity_id_to_label
entity_labels_df = pd.DataFrame.from_dict(entity_labels, orient='index', columns=['entity_label'])

relation_labels = triples_factory_s.relation_id_to_label
relation_labels_df = pd.DataFrame.from_dict(relation_labels, orient='index', columns=['relation_label'])

#clustering of entities
entity_embeddings = pipeline_result_s.model.entity_embeddings._embeddings

embedding_list = []
for i in range(int(float(entity_embeddings.num_embeddings))):
    vector = (entity_embeddings(torch.LongTensor([i])).tolist()[0])
    embedding_list.append(vector)

#print(embedding_list)

embedding_array = np.array(embedding_list)
embedding_tensor = torch.Tensor(embedding_array)

number_of_clusters = number_of_clusters

kmedoids = KMedoids(n_clusters=number_of_clusters, random_state=0)

cluster_labels = kmedoids.fit_predict(embedding_array)

feat_cols = [ 'feature'+ str(i+1) for i in range(entity_embeddings.embedding_dim) ]
cluster_df = pd.DataFrame(embedding_array, columns=feat_cols)
cluster_df['cluster_label'] = cluster_labels

representative_entity_list = []
medoids = kmedoids.cluster_centers_

for index in kmedoids.medoid_indices_:
    entity = index
    label = kmedoids.labels_[index]
    representative_entity_list.append(entity)


ht_list = (list(itertools.combinations(representative_entity_list, 2)))

#assign labels to the representatives
representatives_list_with_labels = []

for ht in ht_list:
    representatives_list_with_labels.append(
        [
        entity_labels_df.iloc[ht[0]][0], 
         entity_labels_df.iloc[ht[1]][0],
            ht[0],
                ht[1]
        ],
        
    )

#perform link prediction task in situation S
dataframes_S = []

for i in representatives_list_with_labels:
    pred_df = get_relation_prediction_df(model_s, i[0], i[1], add_novelties=True ,triples_factory=triples_factory_s)
    pred_df['head_label'] = i[0]
    pred_df['tail_label'] = i[1]
    pred_df['head_id'] = i[2]
    pred_df['tail_id'] = i[3]
    dataframes_S.append(pred_df)
    
result_s = pd.concat(dataframes_S)
result_s = result_s.reset_index(drop=True)

#get triples to inspect
triples_not_in_training_s = result_s[result_s['in_training'] == False]

head_id = triples_not_in_training_s['head_id'].to_list()
relation_id = triples_not_in_training_s['relation_id'].to_list()
tail_id = triples_not_in_training_s['tail_id'].to_list()

triples_not_in_training_ids = []

for idx, ele in enumerate(head_id):
    triples_not_in_training_ids.append([ele, relation_id[idx], tail_id[idx]])
    
    
triples_not_in_training_ids_array = np.array(triples_not_in_training_ids)

reduced_list = []
random_relation = sample(relation_id, number_of_relations_to_inspect)

for idx, ele in enumerate(ht_list):
    for i in random_relation:
        reduced_list.append([ele[0], i, ele[1]])


num_relations = dataset.num_relations

reduced_array = np.array(check_and_replace_triple(reduced_list, triples_not_in_training_ids, num_relations))

#iterative model
mapped_triples_training_s = pipeline_result_s.training.mapped_triples
mapped_triples_testing = dataset.testing.mapped_triples

mapped_triples_training_s_array = mapped_triples_training_s.numpy()

length = len(mapped_triples_training_s)*(positive_oversampling/100)

#results from the iterative model
expected_impact_list = []

scores_list_s = []
scores_list_a = []
scores_list_b = []
scores_list_s.append(result_s['score'].tolist())

variance_list_s = []
variance_list_a = []
variance_list_b = []

variance_list_s.append(np.var(result_s['score'].tolist()))

ztest_list_a = []
ztest_list_b = []


for idx, ele in enumerate(tqdm(reduced_array)):
    
    mapped_triples_training_s_array = mapped_triples_training_s.numpy()
    
    #oversample A
    for j in range (int(length)):
        mapped_triples_training_s_array = np.concatenate((mapped_triples_training_s_array, [reduced_array[idx]]))
        
    pt_tensor_from_list = torch.Tensor(mapped_triples_training_s_array)
    mapped_triples_training_plus_representative_triple = pt_tensor_from_list.type(torch.int64)
    
    training_A = TriplesFactory(mapped_triples_training_plus_representative_triple, entity_to_id, relation_to_id)
    testing_A = TriplesFactory(mapped_triples_testing, entity_to_id, relation_to_id)
    
    #train A
    checkpointA = '{} {}.pt'.format(checkpoint_A_name, reduced_array[idx])
    
    pipeline_result_A = pipeline(
    training=training_A,
    testing=testing_A,
    model= KGE_model,
    model_kwargs = model_kwargs,
    training_loop=training_loop,
    optimizer_kwargs=optimizer_kwargs,
    negative_sampler=basic_negative_sampler,
    evaluator=evaluator,
    loss=loss,
    training_kwargs=dict(
        num_epochs=number_of_epochs,
        checkpoint_name=checkpointA,
        checkpoint_frequency=checkpoint_frequency_min),
    random_seed = seed,

    device=device,
    use_tqdm=use_tqdm

    )

    evaluation_summary_a = pipeline_result_A.metric_results
    losses_a = pipeline_result_A.losses

    metric_df_a = evaluation_summary_a.to_df()
    metric_df_a = metric_df_a.loc[(metric_df_a['Metric'] == metric)]
    
    dataframesA = []
    
    
    #get scores A
    model_A = pipeline_result_A.model
    for i in representatives_list_with_labels:
        pred_df = get_relation_prediction_df(model_A, i[0], i[1], add_novelties=True ,triples_factory=pipeline_result_A.training)
        pred_df['head_label'] = i[0]
        pred_df['tail_label'] = i[1]
        pred_df['head_id'] = i[2]
        pred_df['tail_id'] = i[3]
        dataframesA.append(pred_df)

    result_A = pd.concat(dataframesA)
    
    result_A = result_A.reset_index(drop=True)
    triples_not_in_training_A = result_A[result_A['in_training'] == False]
    
    scores_A = triples_not_in_training_A['score']
        
    #oversample + train B
    checkpointB = '{} {}.pt'.format(checkpoint_B_name, reduced_array[idx])
    triple = reduced_array[idx]

    training_B = TriplesFactory(mapped_triples_training_s, entity_to_id, relation_to_id)
    testing_B = TriplesFactory(mapped_triples_testing, entity_to_id, relation_to_id)
    
    pipeline_result_B = pipeline(
    training=training_B,
    testing=testing_B,
    model= KGE_model,
    model_kwargs = model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    loss=loss,
    training_loop=training_loop,
    negative_sampler=MyNegativeSampler,
    negative_sampler_kwargs = dict(triple=triple, percentage=negative_oversampling), 
    evaluator=evaluator,
    training_kwargs=dict(
        num_epochs=number_of_epochs,
        checkpoint_name=checkpointB,
        checkpoint_frequency=checkpoint_frequency_min),
    random_seed = seed,

    device=device,
    use_tqdm=use_tqdm
    )

    evaluation_summary_b = pipeline_result_B.metric_results
    losses_b = pipeline_result_B.losses

    metric_df_b = evaluation_summary_b.to_df()
    metric_df_b = metric_df_b.loc[(metric_df_b['Metric'] == metric)]
    
    
    #get scores B
    dataframesB = []

    model_B = pipeline_result_B.model
    for i in representatives_list_with_labels:
        pred_df = get_relation_prediction_df(model_B, i[0], i[1], add_novelties=True ,triples_factory=pipeline_result_B.training)
        pred_df['head_label'] = i[0]
        pred_df['tail_label'] = i[1]
        pred_df['head_id'] = i[2]
        pred_df['tail_id'] = i[3]
        dataframesB.append(pred_df)
    
    result_B = pd.concat(dataframesB)
    
    result_B = result_B.reset_index(drop=True)
    
    triples_not_in_training_B = result_B[result_B['in_training'] == False]
    
    
    # getting the Kendall Tau values + p-value
    triples_not_in_training_s_copy = triples_not_in_training_s.copy()
    triples_not_in_training_s_copy.drop(triples_not_in_training_s_copy[(triples_not_in_training_s_copy['head_id'] == reduced_array[idx][0]) & 
        (triples_not_in_training_s_copy['relation_id'] == reduced_array[idx][1]) & 
        (triples_not_in_training_s_copy['tail_id'] == reduced_array[idx][2])].index, inplace = True)
    

    tauA, p_valueA = stats.kendalltau(triples_not_in_training_s_copy['score'], triples_not_in_training_A['score'])
    tauB, p_valueB = stats.kendalltau(triples_not_in_training_s['score'], triples_not_in_training_B['score'])

    #getting the probabilities of a triple being true in situation s
    
    prob_triple_s = triples_not_in_training_s.loc[
        (triples_not_in_training_s['head_id'] == reduced_array[idx][0]) & 
        (triples_not_in_training_s['relation_id'] == reduced_array[idx][1]) & 
        (triples_not_in_training_s['tail_id'] == reduced_array[idx][2])]
    
    prob_triple_s_score = prob_triple_s['score']
    prob_triple_positive_s = prob_triple_s_score.values[0]
    prob_triple_negative_s = (1 - prob_triple_positive_s)

    #getting the probabilities of a triple being true after positively oversampling the triple

    prob_a = result_A.loc[(result_A['head_id'] == reduced_array[idx][0]) & 
        (result_A['relation_id'] == reduced_array[idx][1]) & 
        (result_A['tail_id'] == reduced_array[idx][2])]
    
    prob_triple_a_score = prob_a['score']
    prob_triple_positive_a = prob_triple_a_score.values[0]
    prob_triple_negative_a = (1 - prob_triple_positive_a)

    #getting the probabilities of a triple being true after negatively oversampling the triple

    prob_b = result_B.loc[(result_B['head_id'] == reduced_array[idx][0]) & 
        (result_B['relation_id'] == reduced_array[idx][1]) & 
        (result_B['tail_id'] == reduced_array[idx][2])]
    
    prob_triple_b_score = prob_b['score']
    prob_triple_positive_b = prob_triple_b_score.values[0]
    prob_triple_negative_b = (1 - prob_triple_positive_a)
    
    #expected impact/interestingness
    Expected_impact = ((prob_triple_positive_s*tauA) + (prob_triple_negative_s*tauB))
    
    #results
    expected_impact_list.append([
        
        'triple: ', reduced_array[idx].tolist(),
        'prob-s: ', prob_triple_positive_s,
        'prob-after-a: ', prob_triple_positive_a,
        'prob-after-b: ', prob_triple_positive_b,
        'tauA: ', tauA, 
        'p-valueA: ', p_valueA,
        'tauB: ', tauB,
        'p-valueB: ', p_valueB,
        'expected_impact: ', Expected_impact,
        'metric_df_a: ', metric_df_a.to_dict(),
        'metric_df_b: ', metric_df_b.to_dict(),
        ])

    scores_list_a.append(result_A['score'].tolist())
    scores_list_b.append(result_B['score'].tolist())

    variance_list_a.append(np.var(result_A['score'].tolist()))
    variance_list_b.append(np.var(result_B['score'].tolist()))

    ztest_list_a.append(ztest(result_s['score'].tolist(), result_A['score'].tolist(), value=0))
    ztest_list_b.append(ztest(result_s['score'].tolist(), result_B['score'].tolist(), value=0))



print(' ')
print('exp {}, {} Model with {} epochs {}% positive oversampling, {}% negative oversampling, {} clusters, {} relations per representative h-t, 0.0001LR,  ran on {}'.format(experiment_number, dataset, number_of_epochs, positive_oversampling, negative_oversampling, number_of_clusters, number_of_relations_to_inspect, device))
print(' ')
print('seed: {}'.format(seed))

print('------------------------')
print('------------------------')

print("these are the coordinates for the medoids: \n", medoids)
print(" ")
print("these are the representative entities: \n", representative_entity_list)

print('these are the h-t combinations to test: \n', ht_list)

print('length of all triples that could be inspected: ', len(triples_not_in_training_ids_array))
print(' ')
print('triples that could be inspected:\n', triples_not_in_training_ids_array)

print('------------------------')


print('length of triples to inspect: ', len(reduced_array))
print(' ')
print('triples array to inspect:\n', reduced_array)

print('------------------------')

print('mapped triples of training result from first pipeline(s): ')
print(mapped_triples_training_s)
print('length of mapped triples: ', len(mapped_triples_training_s))
print(' ')    
print('amount of positive oversampling: ', float(positive_oversampling))
print(' ')
print('amount of negative oversampling: ', float(negative_oversampling))
print(' ')
print('array with triples to be tested: ')
print(reduced_array)

print('------------------------')

print(' ')
print('expected impact list: ', expected_impact_list)
print(' ')

print('evaluation summary pipeline s: ')
print(' ')
print(evaluation_summary_s.to_df().to_string())
print(' ')
print('losses s: ', losses_s)
print(' ')

rounded_scores_s = [round(num, 2) for num in scores_list_s[0]]
rounded_scores_a = round_list(scores_list_a)
rounded_scores_b = round_list(scores_list_b)

print('scores to show distribution')
print(scores_list_s)
print('-------------------------')
print(scores_list_a)
print('-------------------------')
print(scores_list_b)
print('-------------------------')

print('variance s: ')
print(variance_list_s)
print('------------------')
print('variance a: ')
print(variance_list_a)
print('------------------')
print('variance b: ')
print(variance_list_b)
print('------------------')

print('z-scores + p-values a: ')
print(ztest_list_a)
print('------------------')
print('z-scores + p-values b: ')
print(ztest_list_b)
print('------------------')


end = time.time()
total_time = end - start
print("\n total time(s): "+ str(total_time))