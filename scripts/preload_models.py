#!/usr/bin/env python3
"""
Preload and verify model weights.

Usage:
    python scripts/preload_models.py
    python scripts/preload_models.py --verify

This script loads all model weights and optionally runs verification predictions.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from app.models.registry import ModelRegistry
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def preload_models():
    """Load all models into registry."""
    registry = ModelRegistry()
    await registry.initialize()

    logger.info("Model Registry Status:")
    status = registry.get_status()

    for name, info in status["models"].items():
        state = "LOADED" if info["loaded"] else "NOT LOADED"
        logger.info(f"  {name}: {state} (v{info['version']})")

    return registry


async def verify_predictions(registry: ModelRegistry):
    """Run verification predictions with synthetic data."""
    logger.info("Running verification predictions...")

    # Create synthetic test data
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days)

    # Generate realistic price movement
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.cumprod(1 + returns)

    test_data = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, n_days)
    })

    # Test each model
    results = {}

    for name in registry.get_available_models():
        model = registry.get_model(name)

        if not model or not model.is_trained:
            results[name] = {"status": "NOT LOADED"}
            continue

        try:
            prediction = await model.predict(test_data, horizon=5)

            results[name] = {
                "status": "SUCCESS",
                "predictions": prediction["predictions"],
                "final_price": prediction["final_price"],
                "trend": prediction["trend"]
            }

            logger.info(f"  {name}: SUCCESS")
            logger.info(f"    Final price: {prediction['final_price']:.2f}")
            logger.info(f"    Trend: {prediction['trend']}")

        except Exception as e:
            results[name] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"  {name}: FAILED - {e}")

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Preload and verify model weights"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification predictions"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="app/models/weights",
        help="Directory containing model weights"
    )

    args = parser.parse_args()

    # Check weights directory
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        logger.warning(f"Weights directory not found: {weights_dir}")
        logger.info("Creating directory...")
        weights_dir.mkdir(parents=True, exist_ok=True)

    # List available weights
    weight_files = list(weights_dir.glob("*"))
    if weight_files:
        logger.info("Available weight files:")
        for f in weight_files:
            size_kb = f.stat().st_size / 1024
            logger.info(f"  {f.name} ({size_kb:.1f} KB)")
    else:
        logger.warning("No weight files found. Run train_models.py first.")

    # Load models
    registry = await preload_models()

    # Verify if requested
    if args.verify:
        await verify_predictions(registry)

    # Summary
    status = registry.get_status()
    loaded = status["loaded_models"]
    total = status["total_models"]

    if loaded == total:
        logger.info(f"All {total} models loaded successfully!")
        sys.exit(0)
    else:
        logger.warning(f"Only {loaded}/{total} models loaded")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
