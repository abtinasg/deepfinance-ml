#!/usr/bin/env python3
"""
Offline training script for DeepFinance ML models.

Usage:
    python scripts/train_models.py --symbol AAPL --days 365
    python scripts/train_models.py --symbols AAPL,MSFT,GOOGL --days 180

This script trains all models offline and saves weights to disk.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from app.models.xgboost_model import XGBoostPredictor
from app.models.lstm_model import LSTMPredictor
from app.models.transformer_model import TransformerPredictor
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def fetch_training_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch historical data for training."""
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra days for features

    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df.columns = [col.lower() for col in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.reset_index(drop=True)

    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df


def train_xgboost(data: pd.DataFrame, output_path: Path) -> bool:
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")

    model = XGBoostPredictor()

    if model.train(data):
        model.save_weights(str(output_path))
        logger.info(f"XGBoost weights saved to {output_path}")
        return True

    logger.error("XGBoost training failed")
    return False


def train_lstm(data: pd.DataFrame, output_path: Path, epochs: int = 100) -> bool:
    """Train LSTM model."""
    logger.info(f"Training LSTM model for {epochs} epochs...")

    model = LSTMPredictor()

    if model.train(data, epochs=epochs):
        model.save_weights(str(output_path))
        logger.info(f"LSTM weights saved to {output_path}")
        return True

    logger.error("LSTM training failed")
    return False


def train_transformer(data: pd.DataFrame, output_path: Path, epochs: int = 100) -> bool:
    """Train Transformer model."""
    logger.info(f"Training Transformer model for {epochs} epochs...")

    model = TransformerPredictor()

    if model.train(data, epochs=epochs):
        model.save_weights(str(output_path))
        logger.info(f"Transformer weights saved to {output_path}")
        return True

    logger.error("Transformer training failed")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepFinance ML models offline"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to train on (default: SPY for general market)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to train on"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data (default: 365)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for neural networks (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="app/models/weights",
        help="Directory to save weights"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to train: all, xgboost, lstm, transformer"
    )

    args = parser.parse_args()

    # Determine symbols to train on
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = [args.symbol.upper()]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    models_to_train = []
    if args.models == "all":
        models_to_train = ["xgboost", "lstm", "transformer"]
    else:
        models_to_train = [m.strip().lower() for m in args.models.split(",")]

    logger.info(f"Training models: {models_to_train}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days: {args.days}")

    # Combine data from all symbols
    all_data = []

    for symbol in symbols:
        try:
            data = fetch_training_data(symbol, args.days)
            all_data.append(data)
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")

    if not all_data:
        logger.error("No training data available")
        sys.exit(1)

    # Concatenate data (simple approach - could be improved)
    training_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total training samples: {len(training_data)}")

    # Train models
    results = {}

    if "xgboost" in models_to_train:
        results["xgboost"] = train_xgboost(
            training_data,
            output_dir / "xgb.pkl"
        )

    if "lstm" in models_to_train:
        results["lstm"] = train_lstm(
            training_data,
            output_dir / "lstm.pt",
            epochs=args.epochs
        )

    if "transformer" in models_to_train:
        results["transformer"] = train_transformer(
            training_data,
            output_dir / "transformer.pt",
            epochs=args.epochs
        )

    # Summary
    logger.info("=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)

    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{model}: {status}")

    # Return exit code based on results
    if all(results.values()):
        logger.info("All models trained successfully!")
        sys.exit(0)
    else:
        logger.warning("Some models failed to train")
        sys.exit(1)


if __name__ == "__main__":
    main()
