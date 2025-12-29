//! Compatibility facades for sklearn-like API.
//!
//! Provides TSClassifier, TSRegressor, and TSForecaster classes
//! that mirror the Python tsai API.

use burn::prelude::*;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use tsai_core::Seed;
use tsai_data::{train_test_split, TSDataLoaders, TSDataset};

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
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::TSClassifier;
///
/// let mut clf = TSClassifier::new(config);
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// ```
pub struct TSClassifier {
    config: TSClassifierConfig,
    is_fitted: bool,
    n_classes: usize,
    n_vars: usize,
    seq_len: usize,
}

impl TSClassifier {
    /// Create a new classifier.
    pub fn new(config: TSClassifierConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            n_classes: 0,
            n_vars: 0,
            seq_len: 0,
        }
    }

    /// Fit the classifier.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    /// * `y` - Labels of shape (n_samples,) or (n_samples, 1)
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<()> {
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

        // TODO: Create model based on arch, train using Learner
        // This requires backend-specific code

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class labels.
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<i32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted".to_string()));
        }

        // TODO: Run inference
        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, 1)))
    }

    /// Predict class probabilities.
    pub fn predict_proba(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted".to_string()));
        }

        // TODO: Run inference
        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, self.n_classes)))
    }

    /// Get the config.
    pub fn config(&self) -> &TSClassifierConfig {
        &self.config
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
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
pub struct TSRegressor {
    config: TSRegressorConfig,
    is_fitted: bool,
    n_outputs: usize,
}

impl TSRegressor {
    /// Create a new regressor.
    pub fn new(config: TSRegressorConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            n_outputs: 1,
        }
    }

    /// Fit the regressor.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<()> {
        self.n_outputs = y.shape()[1];
        // TODO: Implement training
        self.is_fitted = true;
        Ok(())
    }

    /// Predict values.
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted".to_string()));
        }

        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, self.n_outputs)))
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
pub struct TSForecaster {
    config: TSForecasterConfig,
    is_fitted: bool,
}

impl TSForecaster {
    /// Create a new forecaster.
    pub fn new(config: TSForecasterConfig) -> Self {
        Self {
            config,
            is_fitted: false,
        }
    }

    /// Fit the forecaster.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<()> {
        // TODO: Implement training
        self.is_fitted = true;
        Ok(())
    }

    /// Forecast future values.
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        if !self.is_fitted {
            return Err(TrainError::Other("Model not fitted".to_string()));
        }

        let n_samples = x.shape()[0];
        Ok(Array2::zeros((n_samples, self.config.horizon)))
    }

    /// Get the forecast horizon.
    pub fn horizon(&self) -> usize {
        self.config.horizon
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
