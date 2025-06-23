# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis project: "Informer In Algorithmic Investment Strategies on High Frequency Bitcoin Data". The project implements a comprehensive framework for training time series forecasting models (specifically the Informer transformer) on Bitcoin price data and evaluating investment strategies.

## Architecture

The codebase is organized into three main modules:

- **`src/informer/`**: Custom Informer transformer implementation with attention mechanisms, encoder/decoder architecture, and embedding layers
- **`src/ml/`**: Machine learning utilities including data handling, loss functions (custom GMADL loss), and model factory
- **`src/strategy/`**: Investment strategy implementations (Buy & Hold, MACD, RSI, ML-based), evaluation framework, and performance metrics

## Key Dependencies

- `pytorch-forecasting==1.0.0` - Core time series forecasting framework
- `wandb==0.17.7` - Experiment tracking and dataset management
- `TA-lib==0.4.32` - Technical analysis indicators
- `plotly==5.22.0` - Visualization
- `lightning.pytorch` - Model training framework

## Common Commands

### Installation
```bash
pip install -e .
```

### Training Models
```bash
# Basic training
python scripts/train.py configs/experiments/informer-btcusdt-15m-gmadl.yaml

# With custom parameters
python scripts/train.py configs/experiments/informer-btcusdt-15m-gmadl.yaml \
    --project my-project \
    --seed 42 \
    --patience 10 \
    --store-predictions

# Disable W&B for testing
python scripts/train.py configs/experiments/informer-btcusdt-15m-gmadl.yaml --no-wandb
```

### Testing
```bash
pytest
```

## Configuration Structure

Experiments are defined in YAML files with these key sections:
- `data`: Dataset configuration (W&B dataset name, validation split)
- `fields`: Time series field definitions (target, features, categorical variables)
- `model`: Model architecture parameters (Informer or TemporalFusionTransformer)
- `loss`: Loss function configuration (GMADL, Quantile, RMSE)
- Training parameters: `batch_size`, `max_epochs`, `past_window`, `future_window`

## Data Management

- All datasets are stored and versioned in Weights & Biases
- Dataset format: `"btc-usdt-{timeframe}:latest"` (e.g., "btc-usdt-15m:latest")
- Supported timeframes: 1m, 5m, 15m, 30m
- Features include OHLCV data, technical indicators (MACD, RSI, Bollinger Bands), and external factors (VIX, Fear & Greed Index)
- **Original Paper Resources**: Access to public W&B project with original results, datasets, and model checkpoints: https://wandb.ai/filipstefaniuk/wne-masters-thesis-testing

## Evaluation Framework

The project includes comprehensive strategy evaluation:
- Multiple loss functions for different forecasting objectives
- Strategy parameter sweeps for optimization
- Performance metrics: Sharpe ratio, maximum drawdown, information ratio
- Visualization tools for strategy comparison

## Important Notes

- The project uses Lightning for model training with early stopping and model checkpointing
- All experiments should be tracked in W&B unless using `--no-wandb` flag
- Configuration files are organized by experiment type: `configs/experiments/`, `configs/evaluations/`, `configs/sweeps/`
- Jupyter notebooks in `notebooks/` contain detailed analysis and visualizations
- **Current Focus**: The project is currently focused solely on GMADL loss function with 5-minute Bitcoin data timeframe

## Research Paper Reference

- Always refer to this file when I say refer to the research paper