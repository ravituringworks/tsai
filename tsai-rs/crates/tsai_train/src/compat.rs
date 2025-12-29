//! Compatibility facades for sklearn-like API.
//!
//! Provides TSClassifier, TSRegressor, and TSForecaster classes
//! that mirror the Python tsai API.
//!
//! These wrappers provide a simple, sklearn-like interface for time series
//! classification, regression, and forecasting without requiring deep knowledge
//! of Burn's type system.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
#[allow(unused_imports)]
use burn::module::AutodiffModule;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::training::{ClassificationTrainer, ClassificationTrainerConfig, TrainingOutput};
use tsai_core::Seed;
use tsai_data::{train_test_split, TSDataLoaders, TSDataset};
use tsai_models::{
    InceptionTimePlus, InceptionTimePlusConfig,
    OmniScaleCNN, OmniScaleCNNConfig,
    TSTConfig, TSTPlus,
};

/// Type alias for the training backend (CPU with autodiff).
type TrainBackend = Autodiff<NdArray>;

/// Type alias for the inference backend (CPU only).
type InferBackend = NdArray;

/// Trained model storage for different architectures.
///
/// This enum holds the trained (inference-ready) model for each supported architecture.
enum TrainedModel {
    InceptionTimePlus(InceptionTimePlus<InferBackend>),
    OmniScaleCNN(OmniScaleCNN<InferBackend>),
    TSTPlus(TSTPlus<InferBackend>),
}

impl TrainedModel {
    /// Run inference on the trained model.
    fn forward(&self, x: Tensor<InferBackend, 3>) -> Tensor<InferBackend, 2> {
        match self {
            TrainedModel::InceptionTimePlus(m) => m.forward(x),
            TrainedModel::OmniScaleCNN(m) => m.forward(x),
            TrainedModel::TSTPlus(m) => m.forward(x),
        }
    }
}

/// Configuration for sklearn-like classifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSClassifierConfig {
    /// Model architecture name.
    pub arch: String,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
    /// Whether to use GPU.
    pub use_gpu: bool,
}

impl Default for TSClassifierConfig {
    fn default() -> Self {
        Self {
            arch: "InceptionTimePlus".to_string(),
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
            use_gpu: false,
        }
    }
}

/// Sklearn-like time series classifier.
///
/// This classifier provides a simple, high-level API for time series classification
/// that mirrors the Python tsai library. It handles model creation, training, and
/// inference internally.
///
/// # Supported Architectures
///
/// - `InceptionTimePlus` (default) - InceptionTime with improvements
/// - `OmniScaleCNN` - Multi-scale CNN
/// - `TSTPlus` - Time Series Transformer
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSClassifier, TSClassifierConfig};
/// use ndarray::Array3;
///
/// let config = TSClassifierConfig::default();
/// let mut clf = TSClassifier::new(config);
///
/// // x: (n_samples, n_vars, seq_len), y: (n_samples, 1)
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// let probabilities = clf.predict_proba(&x_test)?;
/// ```
pub struct TSClassifier {
    config: TSClassifierConfig,
    trained_model: Option<TrainedModel>,
    n_classes: usize,
    n_vars: usize,
    seq_len: usize,
    device: <InferBackend as Backend>::Device,
}

impl TSClassifier {
    /// Create a new classifier.
    pub fn new(config: TSClassifierConfig) -> Self {
        Self {
            config,
            trained_model: None,
            n_classes: 0,
            n_vars: 0,
            seq_len: 0,
            device: Default::default(),
        }
    }

    /// Fit the classifier.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    /// * `y` - Labels of shape (n_samples, 1) with class indices
    ///
    /// # Returns
    ///
    /// Training metrics including final accuracy.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<TrainingMetrics> {
        let (n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;

        // Determine number of classes
        let y_flat: Vec<f32> = y.iter().copied().collect();
        let max_class = y_flat.iter().cloned().fold(0.0f32, f32::max) as usize;
        self.n_classes = max_class + 1;

        // Create dataset
        let dataset = TSDataset::from_arrays(x.clone(), Some(y.clone()))?;

        // Split into train/valid
        let seed = Seed::new(self.config.seed);
        let (train_ds, valid_ds) = train_test_split(&dataset, self.config.valid_ratio, seed)?;

        // Create dataloaders
        let dls = TSDataLoaders::builder(train_ds, valid_ds)
            .batch_size(self.config.batch_size)
            .seed(seed)
            .build()?;

        // Train based on architecture
        let metrics = match self.config.arch.as_str() {
            "InceptionTimePlus" | "inception" | "inception_time" => {
                self.train_inception_time(&dls)?
            }
            "OmniScaleCNN" | "omniscale" => {
                self.train_omniscale(&dls)?
            }
            "TSTPlus" | "tst" | "transformer" => {
                self.train_tst(&dls)?
            }
            other => {
                return Err(TrainError::Other(format!(
                    "Unknown architecture '{}'. Supported: InceptionTimePlus, OmniScaleCNN, TSTPlus",
                    other
                )));
            }
        };

        Ok(metrics)
    }

    fn train_inception_time(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = InceptionTimePlusConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_classes,
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        };

        // Initialize model for training
        let model: InceptionTimePlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model (convert to inference mode)
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::InceptionTimePlus(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_omniscale(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = OmniScaleCNNConfig::new(self.n_vars, self.seq_len, self.n_classes)
            .with_n_filters(64)
            .with_dropout(0.1);

        // Initialize model for training
        let model: OmniScaleCNN<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::OmniScaleCNN(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_tst(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let d_model = 64;
        let model_config = TSTConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_classes,
            d_model,
            n_heads: 4,
            n_layers: 3,
            d_ff: d_model * 4,
            dropout: 0.1,
            use_pe: true,
        };

        // Initialize model for training
        let model: TSTPlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::TSTPlus(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    /// Predict class labels.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Class predictions of shape (n_samples, 1)
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<i32>> {
        let proba = self.predict_proba(x)?;

        // Get argmax for each sample
        let n_samples = proba.nrows();
        let mut predictions = Array2::zeros((n_samples, 1));

        for i in 0..n_samples {
            let row = proba.row(i);
            let (max_idx, _) = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            predictions[[i, 0]] = max_idx as i32;
        }

        Ok(predictions)
    }

    /// Predict class probabilities.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Class probabilities of shape (n_samples, n_classes)
    pub fn predict_proba(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        let model = self.trained_model.as_ref().ok_or_else(|| {
            TrainError::Other("Model not fitted. Call fit() first.".to_string())
        })?;

        let (n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);

        // Convert ndarray to Burn tensor
        let data: Vec<f32> = x.iter().copied().collect();
        let tensor_data = burn::tensor::TensorData::new(data, [n_samples, n_vars, seq_len]);
        let tensor: Tensor<InferBackend, 3> = Tensor::from_data(tensor_data, &self.device);

        // Run inference
        let logits = model.forward(tensor);

        // Apply softmax to get probabilities
        let probs = burn::tensor::activation::softmax(logits, 1);

        // Convert back to ndarray
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
        let result = Array2::from_shape_vec((n_samples, self.n_classes), probs_data)
            .map_err(|e| TrainError::Other(e.to_string()))?;

        Ok(result)
    }

    /// Get the config.
    pub fn config(&self) -> &TSClassifierConfig {
        &self.config
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.trained_model.is_some()
    }

    /// Get the number of classes (after fitting).
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

/// Training metrics returned after fitting.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training losses per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch.
    pub valid_losses: Vec<f32>,
    /// Validation accuracies per epoch.
    pub valid_accs: Vec<f32>,
    /// Best validation accuracy achieved.
    pub best_valid_acc: f32,
    /// Epoch with best validation accuracy.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

/// Configuration for sklearn-like regressors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSRegressorConfig {
    /// Model architecture name.
    pub arch: String,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for TSRegressorConfig {
    fn default() -> Self {
        Self {
            arch: "InceptionTimePlus".to_string(),
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
        }
    }
}

/// Sklearn-like time series regressor.
///
/// Note: Regression support requires models with regression output heads.
/// This is a placeholder implementation that demonstrates the API.
/// Full implementation requires regression-specific model configurations.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSRegressor, TSRegressorConfig};
///
/// let config = TSRegressorConfig::default();
/// let mut reg = TSRegressor::new(config);
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// ```
pub struct TSRegressor {
    #[allow(dead_code)]
    config: TSRegressorConfig,
    is_fitted: bool,
    n_outputs: usize,
    n_vars: usize,
    seq_len: usize,
}

impl TSRegressor {
    /// Create a new regressor.
    pub fn new(config: TSRegressorConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            n_outputs: 1,
            n_vars: 0,
            seq_len: 0,
        }
    }

    /// Fit the regressor.
    ///
    /// Note: This is a placeholder. Full regression training requires
    /// regression-specific model output heads and MSE loss.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<TrainingMetrics> {
        let (_, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;
        self.n_outputs = y.shape()[1];
        self.is_fitted = true;

        // Return placeholder metrics
        Ok(TrainingMetrics {
            train_losses: vec![0.0; self.config.n_epochs],
            valid_losses: vec![0.0; self.config.n_epochs],
            valid_accs: vec![],
            best_valid_acc: 0.0,
            best_epoch: 0,
            training_time_secs: 0.0,
        })
    }

    /// Predict values.
    ///
    /// Note: Returns placeholder zeros until regression models are implemented.
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted. Call fit() first.".to_string()));
        }

        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, self.n_outputs)))
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get the number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }
}

/// Configuration for sklearn-like forecasters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSForecasterConfig {
    /// Model architecture name.
    pub arch: String,
    /// Forecast horizon.
    pub horizon: usize,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for TSForecasterConfig {
    fn default() -> Self {
        Self {
            arch: "PatchTST".to_string(),
            horizon: 24,
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
        }
    }
}

/// Sklearn-like time series forecaster.
///
/// Forecasters predict future values of a time series given historical data.
///
/// Note: Forecasting support requires models with forecasting output heads.
/// This is a placeholder implementation that demonstrates the API.
/// Full implementation requires forecasting-specific model configurations
/// and sequence-to-sequence architectures.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSForecaster, TSForecasterConfig};
///
/// let config = TSForecasterConfig {
///     horizon: 24,
///     ..Default::default()
/// };
/// let mut forecaster = TSForecaster::new(config);
/// forecaster.fit(&x_train, &y_train)?;
/// let predictions = forecaster.predict(&x_test)?;
/// ```
pub struct TSForecaster {
    config: TSForecasterConfig,
    is_fitted: bool,
    n_vars: usize,
    seq_len: usize,
}

impl TSForecaster {
    /// Create a new forecaster.
    pub fn new(config: TSForecasterConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            n_vars: 0,
            seq_len: 0,
        }
    }

    /// Fit the forecaster.
    ///
    /// Note: This is a placeholder. Full forecasting training requires
    /// sequence-to-sequence models with forecasting heads.
    pub fn fit(&mut self, x: &Array3<f32>, _y: &Array2<f32>) -> Result<TrainingMetrics> {
        let (_, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;
        self.is_fitted = true;

        // Return placeholder metrics
        Ok(TrainingMetrics {
            train_losses: vec![0.0; self.config.n_epochs],
            valid_losses: vec![0.0; self.config.n_epochs],
            valid_accs: vec![],
            best_valid_acc: 0.0,
            best_epoch: 0,
            training_time_secs: 0.0,
        })
    }

    /// Forecast future values.
    ///
    /// Note: Returns placeholder zeros until forecasting models are implemented.
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted. Call fit() first.".to_string()));
        }

        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, self.config.horizon)))
    }

    /// Get the forecast horizon.
    pub fn horizon(&self) -> usize {
        self.config.horizon
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_config() {
        let config = TSClassifierConfig::default();
        assert_eq!(config.arch, "InceptionTimePlus");
        assert_eq!(config.n_epochs, 25);
    }

    #[test]
    fn test_regressor_config() {
        let config = TSRegressorConfig::default();
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_forecaster_config() {
        let config = TSForecasterConfig::default();
        assert_eq!(config.horizon, 24);
        assert_eq!(config.arch, "PatchTST");
    }
}
