import numpy as np
import pandas as pd
from surprise import SVDpp, Dataset, Reader, NMF
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import KFold
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise import SVD as FunkSVD 
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge 
import os 
import pickle
import copy
import implicit 
import lightgbm as lgb 
import argparse 

base_path = './data/'
train_tbr_path = f'{base_path}train_tbr.csv'
train_ratings_path = f'{base_path}train_ratings.csv'
sample_submission_path = f'{base_path}sample_submission.csv'
output_path = 'submission_stacking_ensemble.csv'
CHECKPOINT_DIR = './checkpoints/' 

def load_data():
    """
    Load and prepare data for training and prediction
    """
    print("Loading data")
    
    train_tbr = pd.read_csv(train_tbr_path) # wishlist data
    print(f"Loaded {len(train_tbr)} wishlist items")
    
    train_ratings = pd.read_csv(train_ratings_path) # ratings data
    print(f"Loaded {len(train_ratings)} ratings")
    
    # Remark that the ratings are provided in the format sid_pid 
    # In order to better handle the statistics over scientists and papers, we split the sid_pid in 2 columns
    train_ratings['sid'] = train_ratings['sid_pid'].apply(lambda x: int(x.split('_')[0])) # scientist_id column
    train_ratings['pid'] = train_ratings['sid_pid'].apply(lambda x: int(x.split('_')[1])) # paper_id column
    
    # Split sid_pid in 2 columns like above; We reconstruct the sid_pid column when submitting
    sample_submission = pd.read_csv(sample_submission_path) # sid_pid pairs for which we want to predict the rating
    sample_submission['sid'] = sample_submission['sid_pid'].apply(lambda x: int(x.split('_')[0])) # scientist_id column
    sample_submission['pid'] = sample_submission['sid_pid'].apply(lambda x: int(x.split('_')[1])) # paper_id column

    return train_tbr, train_ratings, sample_submission

def prepare_surprise_data(ratings_df, tbr_df, include_wishlist = True, wishlist_rating_config = 3.75):
    """
    Convert ratings dataset to Surprise Library format 
    Optional: Include wishlist as implicit feedback in the dataset with default rating wishlist_rating_config

    ratings_df: pandas dataframe containing the ratings provided by the scientists for papers
    tbr_df: pandas dataframe containing the papers on the wishlist of each scientist
    include_wishlist: bool that decides whether to include wishlist as implicit feedback or not
    wishlist_rating_config: default rating to set for wishlist items
    """

    print("Converting data to Surprise format")
    ratings_data = ratings_df[['sid', 'pid', 'rating']].copy()
    
    # Include wishlist data as implicit feedback if specified
    if include_wishlist and tbr_df is not None:
        print(f"Including {len(tbr_df)} wishlist items as implicit feedback with rating {wishlist_rating_config}")
        wishlist_data = tbr_df[['sid', 'pid']].copy()
        wishlist_data['rating'] = wishlist_rating_config # set the rating for wishlist items 
        
        # Combine ratings with wishlist data
        combined_data = pd.concat([ratings_data, wishlist_data])
        combined_data = combined_data.drop_duplicates(subset=['sid', 'pid']) # there shouldn't be any duplicates but we do the sanity check

    else: # if we don't want the wishlist, change nothing
        combined_data = ratings_data
    
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(combined_data[['sid', 'pid', 'rating']], reader) # Surprise format dataset
    
    return dataset, combined_data

def train_svdpp_model(dataset, n_factors = 150, n_epochs = 30, lr_all = 0.005, reg_all = 0.02, random_state_seed = None, checkpoint_path = None):
    """
    Train an SVD++ model on dataset with the given params and return the trained SVD++ model

    dataset: Surprise format dataset
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading SVD++ model from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(dataset, 'build_full_trainset'):
            trainset = dataset.build_full_trainset()
        else:
            trainset = dataset 
        print("SVD++ model loaded")
        return model

    print("Training SVD++ model")
    model = SVDpp(n_factors = n_factors, n_epochs = n_epochs, lr_all = lr_all, reg_all = reg_all, random_state = random_state_seed)
    
    # the Surprise models require the dataset to be in Trainset format
    if hasattr(dataset, 'build_full_trainset'): # if the dataset is in Surprise Dataset format
        trainset = dataset.build_full_trainset()
    else: # if the dataset is already in Trainset format, keep it as it is
        trainset = dataset
        
    model.fit(trainset)
    print("SVD++ model training complete")

    if checkpoint_path:
        print(f"Saving SVD++ model to {checkpoint_path}...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        print("SVD++ model saved")

    return model

def train_nmf_model(dataset, n_factors = 35, n_epochs = 500, random_state_seed = None, checkpoint_path=None):
    """
    Train an NMF model on dataset with the given params and return the trained NMF model

    dataset: Surprise format dataset
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading NMF model from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(dataset, 'build_full_trainset'):
            trainset = dataset.build_full_trainset()
        else:
            trainset = dataset
        print("NMF model loaded")
        return model
        
    print("Training NMF model")
    model = NMF(n_factors = n_factors, n_epochs = n_epochs, random_state = random_state_seed, verbose = False)
    
    if hasattr(dataset, 'build_full_trainset'):
        trainset = dataset.build_full_trainset()
    else:
        trainset = dataset
        
    model.fit(trainset)
    print("NMF model training complete")

    if checkpoint_path:
        print(f"Saving NMF model to {checkpoint_path}...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        print("NMF model saved.")

    return model


def train_knn_model(dataset, num_neighbors = 40, min_num_neighbors = 3, checkpoint_path=None):
    """
    Train a KNN model on dataset with the given params and return the trained KNN model

    dataset: Surprise format dataset
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading KNN model from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        print("KNN model loaded.")
        return model 
    print("Training KNN model")

    sim_options = {'name': 'pearson_baseline', 'user_based': False, 'min_support': min_num_neighbors}
    model = KNNWithMeans(k = num_neighbors, sim_options = sim_options, verbose = False)
    
    if hasattr(dataset, 'build_full_trainset'):
        trainset_knn = dataset.build_full_trainset()
    else:
        trainset_knn = dataset
        
    model.fit(trainset_knn)
    print("KNN model training complete")

    if checkpoint_path:
        print(f"Saving KNN model to {checkpoint_path}...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        print("KNN model saved")

    return model

def analyze_user_item_stats(combined_data):
    """
    Compute user and item statistics for Bayesian confidence weighting.
    We are interested in:
        - average rating and number of ratings for each scientist
        - mean, std, number of ratings, rating range, bayesian average and confidence for each paper
        - global mean
    """
    
    user_stats = combined_data.groupby('sid')['rating'].agg(['mean', 'count']).reset_index() 
    user_stats.columns = ['sid', 'avg_rating', 'rating_count'] # we compute the scientists' stats 
    
    global_mean = combined_data['rating'].mean() # global average rating
    print(f"Global average rating: {global_mean:.4f}")
    
    item_stats = combined_data.groupby('pid').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max']
    }).reset_index()
    item_stats.columns = ['pid', 'rating_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
    item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
    item_stats['rating_range'] = item_stats['max_rating'] - item_stats['min_rating']
    
    C = 5
    item_stats['bayesian_avg'] = (item_stats['avg_rating'] * item_stats['rating_count'] + 
                                         global_mean * C) / (item_stats['rating_count'] + C)   
    # Calculate confidence score based on count and variance
    # Higher confidence for more ratings and lower variance
    item_stats['confidence'] = 1 - (1 / (1 + item_stats['rating_count'])) 
    # Apply confidence penalty for high variability
    item_stats.loc[item_stats['rating_std'] > 0, 'confidence'] *= (1 - 0.2 * (item_stats['rating_std'] / item_stats['rating_std'].max()))


    positive_ratings = combined_data[combined_data['rating'] >= 4]['rating'].count()
    total_ratings = combined_data['rating'].count()
    positive_bias = positive_ratings / total_ratings
    print(f"Positive rating bias: {positive_bias:.4f} ({positive_ratings} / {total_ratings})")
    
    return user_stats, item_stats, global_mean

def create_interaction_matrix(combined_data):
    """
    Creates a binary matrix from the dataframe that encodes 
    whether the scientist i rated paper j or has paper j on wishlist

    The matrix is returned in CSR format required for ALS
    """
    if hasattr(combined_data, 'all_ratings'):
        n_users = combined_data.n_users
        n_items = combined_data.n_items

        interaction = np.zeros((n_users, n_items))
        for sid, pid, _ in combined_data.all_ratings():
            interaction[sid, pid] = 1
        return csr_matrix(interaction)
    else:
        n_users = len(combined_data['sid'].unique())
        n_items = len(combined_data['pid'].unique())
    
        interaction = np.zeros((n_users, n_items))
        for _, row in combined_data.iterrows():
            interaction[int(row['sid']), int(row['pid'])] = 1
        
        return csr_matrix(interaction)

def train_als_model(interaction_matrix, factors = 100, regularization = 0.01, iterations = 15, random_state_seed = None, checkpoint_path=None):
    """
    Train an ALS model on interaction_matrix with the given params and return the trained ALS model

    interaction_matrix: CSR matrix
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading ALS model from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                model = pickle.load(f)
            print("ALS model loaded.")
            return model
        except Exception as e:
             print(f"Warning: Failed to load ALS checkpoint ({e}). Retraining...")

    print("Training ALS")

    try:
        model = implicit.als.AlternatingLeastSquares(factors = factors, regularization = regularization, iterations = iterations, random_state = random_state_seed)
        model.fit(interaction_matrix.T) # implicit library expects item-user matrix for training, so transpose
        print("ALS model training complete.")

        if checkpoint_path:
            print(f"Saving ALS model to {checkpoint_path}...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model, f)
            print("ALS model saved.")
        
        return model
    
    except Exception as e:
        print(f"Error during ALS training: {e}")
        return None
    
def train_stack_ensemble(trainset, interaction_matrix, base_model_configs):
    trained_models = dict()
    if 'svdpp' in base_model_configs:
        trained_models['svdpp'] = train_svdpp_model(trainset, n_factors = base_model_configs['svdpp']['n_factors'], n_epochs = base_model_configs['svdpp']['n_epochs'], lr_all = base_model_configs['svdpp']['lr_all'], reg_all = base_model_configs['svdpp']['reg_all'], random_state_seed = base_model_configs['svdpp']['random_state_seed'])
    
    if 'nmf' in base_model_configs:
        trained_models['nmf'] = train_nmf_model(trainset, n_factors = base_model_configs['nmf']['n_factors'], n_epochs = base_model_configs['nmf']['n_epochs'], random_state_seed = base_model_configs['nmf']['random_state_seed'])

    if 'knn' in base_model_configs:
        trained_models['knn'] = train_knn_model(trainset, num_neighbors = base_model_configs['knn']['num_neighbors'], min_num_neighbors = base_model_configs['knn']['min_num_neighbors'])

    if 'als' in base_model_configs:
        trained_models['als'] = train_als_model(interaction_matrix, base_model_configs['als']['factors'], base_model_configs['als']['regularization'], base_model_configs['als']['iterations'], base_model_configs['als']['random_state_seed'])

    return trained_models


def predict_stack_ensemble(testset, trained_models):
    predictions = []
    for _, row in testset.iterrows():
        sid = int(row['sid'])
        pid = int(row['pid'])
        prediction = {'sid': sid, 'pid': pid}
        if 'rating' in row:
            prediction['rating'] = row['rating']
        if 'svdpp' in trained_models:
            rating_svdpp = trained_models['svdpp'].predict(sid, pid).est
            prediction['pred_svdpp'] = rating_svdpp
        if 'nmf' in trained_models:
            rating_nmf = trained_models['nmf'].predict(sid, pid).est
            prediction['pred_nmf'] = rating_nmf
        if 'knn' in trained_models:
            rating_knn = trained_models['knn'].predict(sid, pid).est
            prediction['pred_knn'] = rating_knn
        if 'als' in trained_models:
            rating_als = trained_models['als'].item_factors[sid].dot(trained_models['als'].user_factors[pid])
            prediction['pred_als'] = rating_als
        predictions.append(prediction)
    return predictions
        

def generate_oof_predictions(explicit_surprise_dataset, interaction_matrix, base_model_configs, n_splits=5, oof_checkpoint_path=None, global_mean=3.5, random_state=42):
    """
    Generates OOF predictions. Trains Surprise models ONLY on explicit ratings dataset.
    Trains ALS on implicit interactions derived from the fold's explicit data.
    """
    if oof_checkpoint_path and os.path.exists(oof_checkpoint_path):
        print(f"Loading OOF predictions from {oof_checkpoint_path}")
        try:
            oof_df = pd.read_pickle(oof_checkpoint_path)
            expected_cols = ['sid', 'pid', 'rating'] + [f'pred_{name}' for name in base_model_configs.keys()]
            if all(col in oof_df.columns for col in expected_cols):
                 print("OOF predictions loaded successfully.")
                 return oof_df
            else:
                 print("Warning: OOF checkpoint columns mismatch. Regenerating")
        except Exception as e:
            print(f"Warning: Failed to load OOF checkpoint ({e}). Regenerating")
    
    kf = KFold(n_splits = n_splits, random_state = random_state, shuffle = True) 
    predictions = [] 

    # Iterate through folds of the dataset
    for fold, (trainset, testset) in enumerate(kf.split(explicit_surprise_dataset)):
        print(f"Processing Fold {fold+1}/{n_splits}")

        interaction_matrix = create_interaction_matrix(trainset)
        fold_models = train_stack_ensemble(trainset, interaction_matrix, base_model_configs)

        testset_df = pd.DataFrame(testset)
        testset_df.columns = ['sid', 'pid', 'rating']

        fold_predictions = predict_stack_ensemble(testset_df, fold_models)
        predictions = predictions + fold_predictions
    predictions_df = pd.DataFrame(predictions)

    if predictions_df.isnull().any().any():
        predictions_df.fillna(global_mean, inplace = True)

    if oof_checkpoint_path:
        print(f"Saving OOF predictions to {oof_checkpoint_path}...")
        try:
            os.makedirs(os.path.dirname(oof_checkpoint_path), exist_ok=True)
            predictions_df.to_pickle(oof_checkpoint_path)
            print("OOF predictions saved.")
        except Exception as e:
            print(f"Error saving OOF predictions: {e}")
    return predictions_df
    
def train_meta_model(predictions_df, meta_model_params, checkpoint_path = None):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained meta-model from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                meta_model = pickle.load(f)
            print("Meta-model loaded successfully")
            return meta_model
        except Exception as e:
            print(f"Warning: Failed to load meta-model checkpoint ({e}). Retraining")

    feature_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
    X_meta = predictions_df[feature_cols]
    y_meta = predictions_df['rating']

    # meta_model = Ridge(alpha = alpha, random_state = 42)
    meta_model = lgb.LGBMRegressor(**meta_model_params, verbose=-1)
    meta_model.fit(X_meta, y_meta)

    if checkpoint_path:
        print(f"Saving meta-model to {checkpoint_path}...")
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(meta_model, f)
            print("Meta-model saved.")
        except Exception as e:
            print(f"Error saving meta-model: {e}")

    return meta_model

def generate_stacked_submission(full_explicit_dataset, interaction_matrix, base_model_configs, meta_model, sample_submission_df, global_mean, output_path_base):
    trained_models = train_stack_ensemble(full_explicit_dataset, interaction_matrix, base_model_configs)
    
    predictions = predict_stack_ensemble(sample_submission_df, trained_models)
    predictions_df = pd.DataFrame(predictions) 
    if predictions_df.isnull().any().any():
        predictions_df.fillna(global_mean, inplace = True)
    
    feature_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
    final_predictions = meta_model.predict(predictions_df[feature_cols])
    final_predictions = np.clip(final_predictions, 1, 5)

    predictions_df['rating'] = final_predictions
    predictions_df['sid_pid'] = sample_submission_df['sid_pid']
    
    # Include ablation_config_name in the output file path
    ablation_config_name = base_model_configs.get('config_name', 'unknown_config')
    random_seed = base_model_configs.get('random_seed', 'seed')
    final_output_path = output_path_base[:-4] + f'_stacking_{ablation_config_name}_{random_seed}.csv'
    predictions_df[['sid_pid', 'rating']].to_csv(final_output_path, index=False)
    print(f"Submission saved to {final_output_path}")
    return predictions_df


def main():
    parser = argparse.ArgumentParser(description='Run stacking ensemble with optional ablations.')
    parser.add_argument('--ablation_config', type=str, default='all', 
                        choices=['all', 'no_svdpp', 'no_nmf', 'no_knn', 'no_als'],
                        help='Specify which ablation configuration to run.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    args = parser.parse_args()
    ablation_config_name = args.ablation_config
    random_seed = args.random_seed

    BASE_DATA = {
        'wishlist_rating': 1.0,
    }

    BASE_STACKING = {
        'n_splits': 5,
        'meta_model_type': 'lightgbm',
        'meta_model_params': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'num_leaves': 20,
            'random_state': random_seed
        },
    }

    BASE_MODELS = {
        'svdpp': {
            'n_factors': 150, 'n_epochs': 50, 'lr_all': 0.005,
            'reg_all': 0.02, 'random_state_seed': random_seed
        },
        'nmf': {
            'n_factors': 70, 'n_epochs': 550, 'random_state_seed': random_seed
        },
        'knn': {
            'num_neighbors': 50, 'min_num_neighbors': 5,
            'sim_options': {'name': 'pearson_baseline', 'user_based': False, 'min_support': 5}
        },
        'als': {
            'factors': 30, 'regularization': 0.01, 'iterations': 25,
            'random_state_seed': random_seed
        }
    }

    def create_config(name, exclude_models=[]):
        base_models = {k: v for k, v in BASE_MODELS.items() if k not in exclude_models}
        return {
            'config_name': name,
            'data': copy.deepcopy(BASE_DATA),
            'base_models': base_models,
            'stacking': copy.deepcopy(BASE_STACKING),
        }

    CONFIG_MAP = {
        'all': create_config('all'),
        'no_svdpp': create_config('no_svdpp', exclude_models=['svdpp']),
        'no_nmf': create_config('no_nmf', exclude_models=['nmf']),
        'no_knn': create_config('no_knn', exclude_models=['knn']),
        'no_als': create_config('no_als', exclude_models=['als']),
    }

    CONFIG = CONFIG_MAP[ablation_config_name]
    
    CONFIG['base_models']['config_name'] = ablation_config_name 
    CONFIG['base_models']['random_seed'] = random_seed

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_tbr, train_ratings, sample_submission = load_data()

    _, combined_data = prepare_surprise_data( 
        train_ratings, train_tbr,
        include_wishlist = True, 
        wishlist_rating_config= CONFIG['data']['wishlist_rating'] 
    )
    user_stats, item_stats, global_mean = analyze_user_item_stats(combined_data)

    reader = Reader(rating_scale=(1, 5))
    explicit_dataset = Dataset.load_from_df(train_ratings[['sid', 'pid', 'rating']], reader)

    implicit_matrix = create_interaction_matrix(combined_data)

    oof_checkpoint_file = os.path.join(CHECKPOINT_DIR, f'oof_preds_{ablation_config_name}.pkl')
    oof_predictions_df = generate_oof_predictions(
        explicit_dataset, 
        implicit_matrix, 
        CONFIG['base_models'],
        n_splits=CONFIG['stacking']['n_splits'],
        oof_checkpoint_path=oof_checkpoint_file,
        global_mean=global_mean,
        random_state=random_seed
    )

    if oof_predictions_df is None:
        print("Failed to generate OOF predictions. Exiting.")
        return 
    
    meta_model_checkpoint_file = os.path.join(CHECKPOINT_DIR, f'meta_model_{CONFIG["stacking"]["meta_model_type"]}_{ablation_config_name}.pkl')
    trained_meta_model = train_meta_model(
        oof_predictions_df,
        meta_model_params=CONFIG['stacking']['meta_model_params'],
        checkpoint_path=meta_model_checkpoint_file
    )

    if trained_meta_model is None:
        print("Failed to train meta-model. Exiting")
        return

    full_explicit_dataset = explicit_dataset.build_full_trainset()
    print("Generating Final Stacked Submission")

    submission_df = generate_stacked_submission(
        full_explicit_dataset,
        implicit_matrix,
        CONFIG['base_models'],
        trained_meta_model,
        sample_submission,
        global_mean,
        output_path
    )

    if submission_df is not None:
        print("Main script execution complete")
    else:
        print("Submission generation failed")


if __name__ == "__main__":
    main()