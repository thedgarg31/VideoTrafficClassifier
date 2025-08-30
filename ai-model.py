import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ImprovedVideoTrafficClassifier:
    def __init__(self):
        self.rf_model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.feature_names = []
        self.trained = False
        
    def load_data(self, csv_file):
        """Load and basic preprocessing of network traffic data"""
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} network flows")
            print(f"✓ Columns: {list(df.columns)}")
            
            # Basic data quality checks
            print(f"✓ Missing values: {df.isnull().sum().sum()}")
            print(f"✓ Duplicate rows: {df.duplicated().sum()}")
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values more intelligently
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df = df.fillna('UNKNOWN')
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def create_realistic_labels(self, df):
        """
        Create labels using unsupervised clustering + domain expertise
        This avoids the circular logic of rule-based labeling
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler
        
        print("\n=== Creating Realistic Labels ===")
        
        # Use key traffic characteristics for clustering
        cluster_features = ['throughput_bps', 'avg_pkt', 'total_bytes', 'duration', 'pkt_count']
        
        # Handle edge cases
        cluster_data = df[cluster_features].copy()
        
        # Log transform skewed features
        for col in ['throughput_bps', 'total_bytes']:
            cluster_data[col] = np.log1p(cluster_data[col])
        
        # Scale for clustering
        cluster_scaler = MinMaxScaler()
        cluster_scaled = cluster_scaler.fit_transform(cluster_data)
        
        # Use K-means to find natural data groupings
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_scaled)
        
        df['cluster'] = clusters
        
        # Analyze clusters to identify video-like behavior
        print("Cluster Analysis:")
        for i in range(3):
            cluster_data_orig = df[df['cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data_orig)} samples):")
            print(f"  Avg throughput: {cluster_data_orig['throughput_bps'].mean():.0f} bps")
            print(f"  Avg packet size: {cluster_data_orig['avg_pkt'].mean():.1f}")
            print(f"  Avg total bytes: {cluster_data_orig['total_bytes'].mean():.0f}")
            print(f"  Avg duration: {cluster_data_orig['duration'].mean():.2f}s")
            print(f"  Protocols: {cluster_data_orig['app_proto'].value_counts().head(3).to_dict()}")
        
        # Identify the cluster most likely to represent video traffic
        cluster_stats = []
        for i in range(3):
            cluster_subset = df[df['cluster'] == i]
            # Video indicators: high throughput + large packets + substantial data
            video_score = (
                cluster_subset['throughput_bps'].mean() / 1000000 +  # Normalize throughput
                cluster_subset['avg_pkt'].mean() / 1000 +            # Normalize packet size
                np.log1p(cluster_subset['total_bytes'].mean()) / 10   # Normalize bytes
            )
            cluster_stats.append((i, video_score, len(cluster_subset)))
            
        # Sort by video score
        cluster_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nVideo likelihood ranking:")
        for i, (cluster_id, score, count) in enumerate(cluster_stats):
            print(f"  Rank {i+1}: Cluster {cluster_id} (score: {score:.2f}, n={count})")
        
        # Assign labels: top cluster = video, others = non-video
        # But only if the top cluster represents high-throughput traffic
        video_cluster = cluster_stats[0][0]
        video_cluster_data = df[df['cluster'] == video_cluster]
        
        # Validation check: video cluster should have higher than average metrics
        if (video_cluster_data['throughput_bps'].mean() > df['throughput_bps'].quantile(0.7) and
            video_cluster_data['avg_pkt'].mean() > df['avg_pkt'].quantile(0.6)):
            df['is_video'] = (df['cluster'] == video_cluster).astype(int)
        else:
            # Fallback: use percentile-based labeling
            print("Cluster validation failed, using percentile-based labeling")
            high_throughput = df['throughput_bps'] > df['throughput_bps'].quantile(0.8)
            large_packets = df['avg_pkt'] > df['avg_pkt'].quantile(0.75)
            substantial_data = df['total_bytes'] > df['total_bytes'].quantile(0.7)
            https_traffic = df['app_proto'] == 'HTTPS'
            
            df['is_video'] = (high_throughput & large_packets & substantial_data & https_traffic).astype(int)
        
        video_count = df['is_video'].sum()
        total_count = len(df)
        
        print(f"\n✓ Final labeling: {video_count} video ({video_count/total_count*100:.1f}%), "
              f"{total_count-video_count} non-video ({(total_count-video_count)/total_count*100:.1f}%)")
        
        if video_count == 0 or video_count == total_count:
            print("✗ Warning: Imbalanced labeling detected!")
            return None
            
        return df
    
    def engineer_features(self, df):
        """
        Sophisticated feature engineering focusing on behavioral patterns
        """
        print("\n=== Feature Engineering ===")
        
        features = pd.DataFrame()
        
        # 1. BASIC FLOW CHARACTERISTICS
        features['duration_log'] = np.log1p(df['duration'])
        features['pkt_count_log'] = np.log1p(df['pkt_count'])
        features['total_bytes_log'] = np.log1p(df['total_bytes'])
        features['throughput_log'] = np.log1p(df['throughput_bps'])
        
        # 2. PACKET SIZE ANALYSIS
        features['avg_pkt_log'] = np.log1p(df['avg_pkt'])
        features['pkt_size_variance'] = (df['max_pkt'] - df['min_pkt']) / (df['avg_pkt'] + 1)
        features['pkt_size_skew'] = (df['avg_pkt'] - df['median_pkt']) / (df['avg_pkt'] + 1)
        
        # 3. TIMING PATTERNS
        features['mean_iat_log'] = np.log1p(df['mean_iat'])
        features['iat_cv'] = df['std_iat'] / (df['mean_iat'] + 0.001)  # Coefficient of variation
        features['iat_stability'] = 1 / (1 + features['iat_cv'])  # Inverse for stability measure
        
        # 4. TRAFFIC INTENSITY PATTERNS
        features['bytes_per_second'] = df['total_bytes'] / (df['duration'] + 0.001)
        features['packets_per_second'] = df['pkt_count'] / (df['duration'] + 0.001)
        features['bytes_per_packet'] = df['total_bytes'] / (df['pkt_count'] + 1)
        
        # Log transform these as they're highly skewed
        features['bytes_per_second_log'] = np.log1p(features['bytes_per_second'])
        features['packets_per_second_log'] = np.log1p(features['packets_per_second'])
        features['bytes_per_packet_log'] = np.log1p(features['bytes_per_packet'])
        
        # 5. PROTOCOL AND PORT FEATURES
        # One-hot encode protocols (limited to most common)
        protocol_dummies = pd.get_dummies(df['app_proto'], prefix='proto').astype(int)
        # Keep only most frequent protocols to avoid curse of dimensionality
        protocol_counts = df['app_proto'].value_counts()
        top_protocols = protocol_counts.head(5).index
        for proto in top_protocols:
            if f'proto_{proto}' in protocol_dummies.columns:
                features[f'proto_{proto}'] = protocol_dummies[f'proto_{proto}']
        
        # Port analysis
        features['src_port_type'] = pd.cut(df['sport'], bins=[0, 1024, 49152, 65536], 
                                         labels=[0, 1, 2], include_lowest=True).astype(int)
        features['dst_port_type'] = pd.cut(df['dport'], bins=[0, 1024, 49152, 65536], 
                                         labels=[0, 1, 2], include_lowest=True).astype(int)
        
        # Common video/streaming ports
        video_ports = [80, 443, 1935, 8080]  # HTTP, HTTPS, RTMP, Alt-HTTP
        features['uses_video_port'] = df.apply(lambda x: int(x['sport'] in video_ports or x['dport'] in video_ports), axis=1)
        
        # 6. DIRECTIONAL FLOW FEATURES
        features['up_down_ratio_log'] = np.log1p(df['up_down_ratio'])
        
        # 7. ADVANCED STATISTICAL FEATURES
        # Create interaction features for important combinations
        features['throughput_duration_interaction'] = features['throughput_log'] * features['duration_log']
        features['pkt_size_count_interaction'] = features['avg_pkt_log'] * features['pkt_count_log']
        
        # 8. BEHAVIORAL ANOMALY FEATURES
        # Z-scores for detecting unusual behavior
        for col in ['throughput_bps', 'avg_pkt', 'total_bytes']:
            z_scores = np.abs(stats.zscore(df[col]))
            features[f'{col}_zscore'] = z_scores
            features[f'{col}_is_outlier'] = (z_scores > 2).astype(int)
        
        # Replace any infinite or NaN values
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        print(f"✓ Created {len(features.columns)} engineered features")
        
        return features
    
    def select_features(self, X, y, k=20):
        """
        Intelligent feature selection using multiple methods
        """
        print(f"\n=== Feature Selection ===")
        print(f"Starting with {X.shape[1]} features")
        
        # Method 1: Statistical significance (ANOVA F-test)
        selector_stats = SelectKBest(score_func=f_classif, k=min(k*2, X.shape[1]))
        X_stats = selector_stats.fit_transform(X, y)
        selected_features_stats = set(X.columns[selector_stats.get_support()])
        
        # Method 2: Recursive Feature Elimination with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf_selector, n_features_to_select=k, step=1)
        rfe.fit(X, y)
        selected_features_rfe = set(X.columns[rfe.support_])
        
        # Method 3: Feature importance from Random Forest
        rf_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_importance.fit(X, y)
        importance_scores = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_importance.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_important_features = set(importance_scores.head(k)['feature'].tolist())
        
        # Combine methods: take intersection + top important features
        combined_features = selected_features_stats.intersection(selected_features_rfe)
        combined_features = combined_features.union(top_important_features)
        
        # Ensure we have at least k features
        if len(combined_features) < k:
            remaining_features = importance_scores[~importance_scores['feature'].isin(combined_features)]
            additional_features = remaining_features.head(k - len(combined_features))['feature'].tolist()
            combined_features = combined_features.union(set(additional_features))
        
        final_features = list(combined_features)[:k]  # Limit to k features
        
        print(f"✓ Selected {len(final_features)} features using combined methods")
        print("Top 10 selected features:")
        selected_importance = importance_scores[importance_scores['feature'].isin(final_features)].head(10)
        for idx, row in selected_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return X[final_features], final_features
    
    def train_model(self, X, y):
        """
        Train model with proper validation and hyperparameter tuning
        """
        print(f"\n=== Model Training ===")
        
        # Check class distribution
        class_dist = y.value_counts()
        print(f"Class distribution: {dict(class_dist)}")
        
        if len(class_dist) < 2:
            raise ValueError("Need both classes for training!")
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features using robust scaler (handles outliers better)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning with GridSearchCV
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 7, 9, None],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 6, 8],
            'max_features': ['sqrt', 'log2', 0.7]
        }
        
        # Use smaller grid for efficiency if dataset is small
        if len(X_train) < 500:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 7],
                'min_samples_split': [10, 15],
                'min_samples_leaf': [4, 6],
                'max_features': ['sqrt', 0.7]
            }
        
        rf_base = RandomForestClassifier(
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Stratified K-Fold for small datasets
        cv_folds = min(5, len(y_train) // (class_dist.min() * 2))  # Ensure enough samples per fold
        cv = StratifiedKFold(n_splits=max(3, cv_folds), shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            rf_base, 
            param_grid, 
            cv=cv,
            scoring='f1_weighted',  # Better for imbalanced classes
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        self.rf_model = grid_search.best_estimator_
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.rf_model.predict(X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(X_test_scaled)
        
        # Comprehensive evaluation
        test_accuracy = accuracy_score(y_test, y_pred)
        train_pred = self.rf_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"\n=== Model Performance ===")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Overfitting Gap: {train_accuracy - test_accuracy:.4f}")
        
        if y_pred_proba.shape[1] == 2:  # Binary classification
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"ROC AUC: {auc_score:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Video', 'Video']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.trained = True
        self.feature_names = list(X.columns)
        
        return {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': feature_importance,
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy
        }
    
    def plot_comprehensive_results(self, results):
        """
        Create comprehensive visualization of results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature Importance
        top_features = results['feature_importance'].head(15)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_title('Top 15 Feature Importance')
        axes[0, 0].invert_yaxis()
        
        # 2. ROC Curve
        if results['y_pred_proba'].shape[1] == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'][:, 1])
            roc_auc = auc(fpr, tpr)
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        if results['y_pred_proba'].shape[1] == 2:
            precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'][:, 1])
            axes[0, 2].plot(recall, precision, color='blue', lw=2)
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].grid(True)
        
        # 4. Prediction Probability Distribution
        if results['y_pred_proba'].shape[1] == 2:
            video_probs = results['y_pred_proba'][results['y_test'] == 1, 1]
            non_video_probs = results['y_pred_proba'][results['y_test'] == 0, 1]
            
            axes[1, 0].hist(non_video_probs, bins=20, alpha=0.7, label='Non-Video', color='blue', density=True)
            axes[1, 0].hist(video_probs, bins=20, alpha=0.7, label='Video', color='red', density=True)
            axes[1, 0].set_xlabel('Prediction Probability (Video)')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Prediction Probability Distribution')
            axes[1, 0].legend()
        
        # 5. Confusion Matrix Heatmap
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xticklabels(['Non-Video', 'Video'])
        axes[1, 1].set_yticklabels(['Non-Video', 'Video'])
        
        # 6. Model Performance Summary
        axes[1, 2].text(0.1, 0.8, f"Test Accuracy: {results['test_accuracy']:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f"Train Accuracy: {results['train_accuracy']:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"Overfitting Gap: {results['train_accuracy'] - results['test_accuracy']:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
        
        if results['y_pred_proba'].shape[1] == 2:
            auc_score = roc_auc_score(results['y_test'], results['y_pred_proba'][:, 1])
            axes[1, 2].text(0.1, 0.5, f"ROC AUC: {auc_score:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
        
        axes[1, 2].text(0.1, 0.3, f"Features Used: {len(self.feature_names)}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('Model Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, new_df):
        """
        Predict on new network traffic data
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        # Engineer same features as training
        new_features = self.engineer_features(new_df)
        
        # Select same features used in training
        new_features_selected = new_features[self.feature_names]
        
        # Scale features
        new_features_scaled = self.scaler.transform(new_features_selected)
        
        # Make predictions
        predictions = self.rf_model.predict(new_features_scaled)
        prediction_proba = self.rf_model.predict_proba(new_features_scaled)
        
        return predictions, prediction_proba


def main():
    """
    Main execution with comprehensive pipeline
    """
    print("🚀 Advanced Video Traffic Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = ImprovedVideoTrafficClassifier()
    
    # Load data
    df = classifier.load_data('data.csv')
    if df is None:
        return
    
    # Create realistic labels
    df_labeled = classifier.create_realistic_labels(df)
    if df_labeled is None:
        print("✗ Failed to create balanced labels")
        return
    
    # Engineer features
    features = classifier.engineer_features(df_labeled)
    target = df_labeled['is_video']
    
    # Feature selection
    features_selected, selected_feature_names = classifier.select_features(features, target, k=15)
    
    # Train model
    results = classifier.train_model(features_selected, target)
    
    # Visualize results
    classifier.plot_comprehensive_results(results)
    
    # Example predictions
    print(f"\n=== Example Predictions ===")
    sample_data = df_labeled.head(5)
    preds, probs = classifier.predict_new_data(sample_data)
    
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        row = sample_data.iloc[i]
        print(f"Flow {i+1}: {row['src']} -> {row['dst']}")
        print(f"  Protocol: {row['app_proto']}, Bytes: {row['total_bytes']}, Avg Pkt: {row['avg_pkt']:.1f}")
        print(f"  Prediction: {'🎥 Video' if pred == 1 else '📄 Non-Video'} (confidence: {prob.max():.3f})")
        print()
    
    print("✅ Analysis complete!")
    return classifier

if __name__ == "__main__":
    classifier = main()