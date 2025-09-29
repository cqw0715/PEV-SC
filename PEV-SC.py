import os
import pickle
import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Reshape, Multiply, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression 

# ---------------------- 1. Data Loading ----------------------
def load_data(csv_path):
    """Load CSV data and check class distribution"""
    data = pd.read_csv(csv_path)
    print(f"Data distribution:\n{data['label'].value_counts()}")
    sequences = data['sequence'].values
    labels = data['label'].values
    assert len(sequences) == len(labels), "Mismatch between sequence and label counts"
    assert all(isinstance(seq, str) for seq in sequences), "Non-string sequences found"
    return sequences, labels

# ---------------------- 2. ESM-2 Feature Extraction ----------------------
class ESMFeatureExtractor:

    def __init__(self):
        self.model, self.batch_converter, self.device = self._load_model()
        
    def _load_model(self):
        try:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            device = torch.device('cpu')
            model = model.to(device)
            batch_converter = alphabet.get_batch_converter()
            print("Model loaded successfully")
            return model, batch_converter, device
        except Exception as e:
            print(f"Loading failed: {e}")
            raise RuntimeError(f"Failed to load ESM-2 model: {e}")

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
            
        print("Starting feature extraction...")
        features = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
            
            try:
                _, _, batch_tokens = self.batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                seq_lengths = (batch_tokens != self.model.alphabet.padding_idx).sum(1)
                for seq_idx in range(token_representations.size(0)):
                    seq_len = seq_lengths[seq_idx].item()
                    seq_rep = token_representations[seq_idx, :seq_len]
                    features.append(seq_rep.mean(0).cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
                
            if (i//batch_size) % 10 == 0:
                print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)}")
        
        features = np.array(features)
        
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Features cached to: {cache_path}")
            
        return features

# ==================== SE Attention Module ====================
def se_attention_block(input_tensor, reduction_ratio=16):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    channels = input_tensor.shape[channel_axis]
    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(channels // reduction_ratio, activation='relu', use_bias=False)(squeeze)
    excitation = Dense(channels, activation='sigmoid', use_bias=False)(excitation)
    if channel_axis == 1:
        excitation = Reshape((channels, 1))(excitation)
    else:
        excitation = Reshape((1, channels))(excitation)
    scaled = Multiply()([input_tensor, excitation])
    return scaled

# ==================== CBAM Module (Keras version) ====================
def channel_attention(input_tensor, ratio=16):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    channels = input_tensor.shape[channel_axis]
    shared_layer_one = Dense(channels//ratio, activation='relu', use_bias=False)
    shared_layer_two = Dense(channels, use_bias=False)
    avg_pool = GlobalAveragePooling1D()(input_tensor)
    avg_pool = Reshape((1, channels))(avg_pool) if channel_axis == -1 else Reshape((channels, 1))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling1D()(input_tensor)
    max_pool = Reshape((1, channels))(max_pool) if channel_axis == -1 else Reshape((channels, 1))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
    channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
    return Multiply()([input_tensor, channel_attention])

def spatial_attention(input_tensor, kernel_size=7):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=channel_axis, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=channel_axis, keepdims=True))(input_tensor)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    spatial_attention = Conv1D(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', use_bias=False)(concat)
    return Multiply()([input_tensor, spatial_attention])

def cbam_block(input_tensor, ratio=16, kernel_size=7):
    x = channel_attention(input_tensor, ratio)
    x = spatial_attention(x, kernel_size)
    return x

# ---------------------- 3. Model Building ----------------------
def build_se_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(input_layer)
    x = Conv1D(1024, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(512, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = se_attention_block(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = se_attention_block(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = se_attention_block(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    x = Dense(1024, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

def build_cbam_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(input_layer)
    x = Conv1D(1024, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(512, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = cbam_block(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = cbam_block(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = cbam_block(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    x = Dense(1024, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model
    
# ---------------------- 4. Evaluation Functions ----------------------
def evaluate_ensemble(y_true, y_pred_prob):
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, y_pred_class),
        'Sensitivity': recall_score(y_true, y_pred_class),
        'Specificity': recall_score(y_true, y_pred_class, pos_label=0),
        'MCC': matthews_corrcoef(y_true, y_pred_class),
        'F1': f1_score(y_true, y_pred_class),
        'AUC': roc_auc_score(y_true, y_pred_prob)
    }

def print_metrics(metrics, title="Results"):
    print(f"\n{title}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

def print_cv_metrics(metrics_list, title="Cross-validation Results"):
    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    print(f"\n{title} (Mean ± Std):")
    for k in mean_metrics.index:
        print(f"  {k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

# ---------------------- 5. Main Program ----------------------
if __name__ == "__main__":
    
    SEED = 42  # Random seed
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Dataset paths
    TRAIN_CSV = "train_3000_2.csv"
    VAL_CSV = "val_176_2.csv"
    
    # Load and extract features
    feature_extractor = ESMFeatureExtractor()
    train_sequences, train_labels = load_data(TRAIN_CSV)
    val_sequences, val_labels = load_data(VAL_CSV)
    
    train_features = feature_extractor.extract_features(train_sequences, cache_path='train_esm2_features.pkl', batch_size=1)
    val_features = feature_extractor.extract_features(val_sequences, cache_path='val_esm2_features.pkl', batch_size=1)
    
    assert train_features.shape[1] == val_features.shape[1], "Feature dimension mismatch!"

    # Define model builders
    models_to_ensemble = {
        'SE_Model': build_se_model,
        'CBAM_Model': build_cbam_model,
    }
    
    # Initialize K-fold cross-validation
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    # Initialize lists to store metrics for each fold
    soft_voting_metrics = []
    weighted_avg_metrics = []
    stacking_metrics = []

    print("\n" + "="*50)
    print(f"Starting {NUM_FOLDS}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
        
        X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
        y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
        class_weights = {i: w for i, w in enumerate(class_weights)}
        
        # Train base models and collect predictions
        fold_predictions = {}
        fold_aucs = {}
        for name, builder in models_to_ensemble.items():
            print(f"Training base model: {name}...")
            model = builder(X_train_fold.shape[1])
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=0
            )
            
            pred_prob = model.predict(X_val_fold).flatten()
            fold_predictions[name] = pred_prob
            fold_aucs[name] = roc_auc_score(y_val_fold, pred_prob)
        
        # ---------------------- Stacking ----------------------
        stacked_features = np.array(list(fold_predictions.values())).T
        meta_model = LogisticRegression(solver='liblinear')
        meta_model.fit(stacked_features, y_val_fold) 
        
        stacking_preds = meta_model.predict_proba(stacked_features)[:, 1]
        stacking_metrics.append(evaluate_ensemble(y_val_fold, stacking_preds))

    # Print cross-validation results
    print("\n" + "="*50)
    print("All folds completed.")
    print_cv_metrics(stacking_metrics, title="Stacking - Cross-validation Results")
    
    # ---------------------- Final Evaluation on Independent Validation Set ----------------------
    print("\n" + "="*50)
    print("Final evaluation on independent validation set")
    
    final_pred_probs = {}
    for name, builder in models_to_ensemble.items():
        model = builder(train_features.shape[1])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        model.fit(
            train_features, train_labels,
            validation_data=(val_features, val_labels),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )
        final_pred_probs[name] = model.predict(val_features).flatten()

    # Stacking
    final_stacked_features = np.array(list(final_pred_probs.values())).T
    final_meta_model = LogisticRegression(solver='liblinear')
    final_meta_model.fit(np.array(list(final_pred_probs.values())).T, val_labels)
    final_stacking_preds = final_meta_model.predict_proba(final_stacked_features)[:, 1]
    print_metrics(evaluate_ensemble(val_labels, final_stacking_preds), title="Final Results - Stacking")