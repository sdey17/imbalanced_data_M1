import logging

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import deepchem as dc
from chembl_webresource_client.new_client import new_client
import requests

from utils import is_valid_structure, standardize_and_canonicalize, computeMorganFP

scaffoldsplitter = dc.splits.ScaffoldSplitter()

'''
Define the range to be used for active/inactive classification
'''

active_range=1
inactive_range=10

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    '''
    1. Collect and combine the data from BindingDB and ChEMBL
    '''

    # ChEMBL
    target = new_client.target
    activity = new_client.activity
    ch_data = pd.DataFrame(activity.filter(target_chembl_id__in=['CHEMBL216']))
    ch_data_filt = ch_data[ch_data['standard_type'].isin(['EC50', 'IC50', 'Ki'])][['canonical_smiles', 'standard_relation', 'standard_units', 'standard_value']]
    # Optimized: single-pass filtering with cached str.contains results
    contains_gt_or_lt = ch_data_filt['standard_relation'].str.contains('>|<')
    chm_absolute = ch_data_filt[~contains_gt_or_lt][['canonical_smiles','standard_value']]
    chm_greater = ch_data_filt[ch_data_filt['standard_relation'].str.contains('>')][['canonical_smiles','standard_value']]
    chm_greater_inactives = chm_greater[chm_greater['standard_value'].astype(float) > (inactive_range*1000)]
    m1_chm = pd.concat([chm_absolute, chm_greater_inactives])
    m1_chm.columns = ['SMILES', 'Value']

    # BindingDB
    url = f"http://bindingdb.org/rest/getLigandsByUniprot?uniprot=P11229;100000&response=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: {response.status_code}")
    bdb_activity = pd.DataFrame.from_dict(data.get('getLindsByUniprotResponse').get('bdb.affinities'))
    bdb_activity_filt = bdb_activity[bdb_activity['bdb.affinity_type'].isin(['EC50', 'IC50', 'Ki'])][['bdb.smile', 'bdb.affinity']]
    # Optimized: single-pass filtering with cached str.contains results
    bdb_contains_gt_or_lt = bdb_activity_filt['bdb.affinity'].str.contains('>|<')
    bdb_absolute = bdb_activity_filt[~bdb_contains_gt_or_lt][['bdb.smile','bdb.affinity']]
    bdb_greater = bdb_activity_filt[bdb_activity_filt['bdb.affinity'].str.contains('>')][['bdb.smile','bdb.affinity']]
    bdb_greater['bdb.affinity'] = bdb_greater['bdb.affinity'].str.split('>').str[1]
    bdb_greater_inactives = bdb_greater[bdb_greater['bdb.affinity'].astype(float) > (inactive_range*1000)]
    m1_bdb = pd.concat([bdb_absolute, bdb_greater_inactives])
    m1_bdb.columns = ['SMILES', 'Value']

    # Optimized: use assign() chaining to reduce DataFrame copies
    comb_all = (pd.concat([m1_chm, m1_bdb])
                .assign(SMILES=lambda df: df['SMILES'].str.split('|').str[0])
                .assign(Value=lambda df: df['Value'].astype(float) / 1000)
                .sort_values(by='Value')
                .reset_index(drop=True))
    print('Original combined data', comb_all.shape[0])

    '''
    2. Standardize the smiles and remove invalid ones
    '''

    comb_all['is_valid'] = comb_all['SMILES'].apply(is_valid_structure)
    df_valid = comb_all[comb_all['is_valid']== True].drop(columns=['is_valid'])
    df_valid['SMILES'] = df_valid['SMILES'].apply(standardize_and_canonicalize)
    print('Original combined data after SMILES standardization', df_valid.shape[0])

    df_valid = df_valid[['SMILES','Value']].reset_index(drop=True)
    df_valid.to_csv('../Data/Input/M1_public_wActivity.csv')

    '''
    3. Label the data as actives and inactives based the range defined earlier
    '''

    df_valid['Activity'] = 2

    df_valid.loc[df_valid['Value'] < active_range, 'Activity'] = 1
    df_valid.loc[df_valid['Value'] > inactive_range, 'Activity'] = 0
    df_valid_filtered = df_valid[df_valid['Activity'] < 2]
    print('Original combined data w/Labels', df_valid_filtered['Activity'].value_counts())

    '''
    4. Remove duplicate and conflicting SMILES; calculate 1024 bit Morgan Fingerprints
    '''

    all = df_valid_filtered.drop_duplicates(subset=['SMILES','Activity'])
    data = all.drop_duplicates(subset=['SMILES'], keep=False).reset_index(drop=True)
    print('Labelled combined data w/o duplicates', data['Activity'].value_counts())

    PandasTools.AddMoleculeColumnToFrame(frame=data, smilesCol='SMILES', molCol='Molecule')
    data['Morgan2FP'] = data['Molecule'].map(computeMorganFP)

    '''
    5. Split the data into actives and inactives
    6. Define the deepchem dataset and use scaffold split for training and test seperately for actives and inactives (due to the data imbalance problem)
    '''

    actives = data[data['Activity'] == 1]
    inactives = data[data['Activity'] == 0]
    acx = actives['Morgan2FP'].to_numpy()
    inx = inactives['Morgan2FP'].to_numpy()
    acy = actives['Activity'].to_numpy()
    iny = inactives['Activity'].to_numpy()

    dataset_ac = dc.data.DiskDataset.from_numpy(X=acx,y=acy,w=np.zeros(len(acx)),ids=actives['SMILES'])
    dataset_in = dc.data.DiskDataset.from_numpy(X=inx,y=iny,w=np.zeros(len(inx)),ids=inactives['SMILES'])
    train_dataset_ac, test_dataset_ac = scaffoldsplitter.train_test_split(dataset_ac, frac_train=0.8, seed = 10)
    train_dataset_in, test_dataset_in = scaffoldsplitter.train_test_split(dataset_in, frac_train=0.8, seed = 10)

    train_dataset_ac_df = pd.DataFrame([train_dataset_ac.ids, train_dataset_ac.X, train_dataset_ac.y], index=['SMILES', 'Morgan2FP','Activity']).T
    train_dataset_in_df = pd.DataFrame([train_dataset_in.ids, train_dataset_in.X, train_dataset_in.y], index=['SMILES', 'Morgan2FP','Activity']).T
    test_dataset_ac_df = pd.DataFrame([test_dataset_ac.ids, test_dataset_ac.X, test_dataset_ac.y], index=['SMILES', 'Morgan2FP','Activity']).T
    test_dataset_in_df = pd.DataFrame([test_dataset_in.ids, test_dataset_in.X, test_dataset_in.y], index=['SMILES', 'Morgan2FP','Activity']).T

    '''
    7. Check for common fingerprints between the active and inactive datasets and remove such entries
    '''

    # Optimized O(n) set-based overlap detection instead of O(n²) nested loops
    inactive_fps = {tuple(fp): i for i, fp in enumerate(train_dataset_in_df['Morgan2FP'])}
    active_fps = {tuple(fp): j for j, fp in enumerate(train_dataset_ac_df['Morgan2FP'])}

    overlapping_fps = set(inactive_fps.keys()) & set(active_fps.keys())
    ind_ac_tr = [active_fps[fp] for fp in overlapping_fps]
    ind_in_tr = [inactive_fps[fp] for fp in overlapping_fps]

    if overlapping_fps:
        for i, j in zip(ind_in_tr, ind_ac_tr):
            print('Train overlap', i, j)

    train_dataset_ac_df = train_dataset_ac_df.drop(ind_ac_tr).reset_index(drop=True)
    train_dataset_in_df = train_dataset_in_df.drop(ind_in_tr).reset_index(drop=True)

    # Optimized O(n) set-based overlap detection for test set
    test_inactive_fps = {tuple(fp): i for i, fp in enumerate(test_dataset_in_df['Morgan2FP'])}
    test_active_fps = {tuple(fp): j for j, fp in enumerate(test_dataset_ac_df['Morgan2FP'])}

    overlapping_test_fps = set(test_inactive_fps.keys()) & set(test_active_fps.keys())
    ind_ac_te = [test_active_fps[fp] for fp in overlapping_test_fps]
    ind_in_te = [test_inactive_fps[fp] for fp in overlapping_test_fps]

    if overlapping_test_fps:
        for i, j in zip(ind_in_te, ind_ac_te):
            print('Test overlap', i, j)

    test_dataset_ac_df = test_dataset_ac_df.drop(ind_ac_te).reset_index(drop=True)
    test_dataset_in_df = test_dataset_in_df.drop(ind_in_te).reset_index(drop=True)

    '''
    8. Concatenate the actives and inactives to form final training and test datasets.
    9. Do a final check between the actives and inactives to remove duplicates
    '''

    training_set = pd.concat([train_dataset_ac_df,train_dataset_in_df]).reset_index(drop=True)
    ext_test_set = pd.concat([test_dataset_ac_df,test_dataset_in_df]).reset_index(drop=True)

    print('Initial Training set',training_set['Activity'].value_counts())

    # Optimized O(n) set-based overlap detection between training and test
    training_fps = {tuple(fp): i for i, fp in enumerate(training_set['Morgan2FP'])}
    test_fps_set = {tuple(fp) for fp in ext_test_set['Morgan2FP']}

    overlapping_train_test = training_fps.keys() & test_fps_set
    source = [training_fps[fp] for fp in overlapping_train_test]

    if overlapping_train_test:
        for fp in overlapping_train_test:
            print('Public overlap', training_fps[fp], 'found in test set')

    training_set = training_set.drop(source).reset_index(drop=True)
    print('Final training set',training_set['Activity'].value_counts())
    print('Ext Test set',ext_test_set['Activity'].value_counts())

    training_set[['SMILES','Activity']].to_csv('../Data/Training/M1/M1_training_set.csv', index=False)
    ext_test_set[['SMILES','Activity']].to_csv('../Data/Test/M1_test_scaffold_split.csv', index=False)

if __name__ == "__main__":
    main()