"""
Model registry for managing ML models and their weights.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.core.logging import get_logger
from app.models.base_model import BasePredictor
from app.models.xgboost_model import XGBoostPredictor
from app.models.lstm_model import LSTMPredictor
from app.models.transformer_model import TransformerPredictor

logger = get_logger(__name__)


class ModelRegistry:
    """
    Central registry for managing ML models.

    Handles model loading, versioning, and lifecycle.
    """

    def __init__(self):
        self._models: dict[str, BasePredictor] = {}
        self._loaded_at: Optional[datetime] = None
        self._weights_dir = Path("app/models/weights")
        self._lock = asyncio.Lock()

    @property
    def models(self) -> dict[str, BasePredictor]:
        """Get all registered models."""
        return self._models

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return len(self._models) > 0 and self._loaded_at is not None

    async def initialize(self) -> None:
        """
        Initialize all models and load pre-trained weights.

        This should be called once at application startup.
        """
        async with self._lock:
            logger.info("Initializing model registry...")

            # Create model instances
            self._models = {
                "xgboost": XGBoostPredictor(),
                "lstm": LSTMPredictor(),
                "transformer": TransformerPredictor()
            }

            # Ensure weights directory exists
            self._weights_dir.mkdir(parents=True, exist_ok=True)

            # Load pre-trained weights
            weights_loaded = await self._load_all_weights()

            self._loaded_at = datetime.utcnow()

            loaded_count = sum(1 for m in self._models.values() if m.is_trained)
            logger.info(
                f"Model registry initialized: {loaded_count}/{len(self._models)} models loaded",
                extra={"models": list(self._models.keys())}
            )

    async def _load_all_weights(self) -> bool:
        """Load weights for all models."""
        weight_files = {
            "xgboost": self._weights_dir / "xgb.pkl",
            "lstm": self._weights_dir / "lstm.pt",
            "transformer": self._weights_dir / "transformer.pt"
        }

        any_loaded = False
        for name, model in self._models.items():
            weight_path = weight_files.get(name)
            if weight_path and weight_path.exists():
                try:
                    if model.load_weights(str(weight_path)):
                        any_loaded = True
                        logger.info(f"Loaded weights for {name}")
                except Exception as e:
                    logger.error(f"Failed to load weights for {name}: {e}")
            else:
                logger.warning(f"No pre-trained weights found for {name}")

        return any_loaded

    async def reload_models(self, model_names: Optional[list[str]] = None) -> dict[str, bool]:
        """
        Reload model weights.

        Args:
            model_names: Specific models to reload (None = all).

        Returns:
            Dictionary of model names to reload success status.
        """
        async with self._lock:
            if model_names is None:
                model_names = list(self._models.keys())

            results = {}
            weight_files = {
                "xgboost": self._weights_dir / "xgb.pkl",
                "lstm": self._weights_dir / "lstm.pt",
                "transformer": self._weights_dir / "transformer.pt"
            }

            for name in model_names:
                if name not in self._models:
                    results[name] = False
                    continue

                model = self._models[name]
                weight_path = weight_files.get(name)

                if weight_path and weight_path.exists():
                    try:
                        results[name] = model.load_weights(str(weight_path))
                    except Exception as e:
                        logger.error(f"Failed to reload {name}: {e}")
                        results[name] = False
                else:
                    results[name] = False

            logger.info(f"Model reload completed: {results}")
            return results

    def get_model(self, name: str) -> Optional[BasePredictor]:
        """
        Get a model by name.

        Args:
            name: Model name (lowercase).

        Returns:
            Model instance or None if not found.
        """
        return self._models.get(name.lower())

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return list(self._models.keys())

    def get_model_info(self, name: str) -> Optional[dict[str, Any]]:
        """Get detailed model information."""
        model = self.get_model(name)
        if not model:
            return None

        settings = get_settings()
        weights = {
            "xgboost": settings.XGBOOST_WEIGHT,
            "lstm": settings.LSTM_WEIGHT,
            "transformer": settings.TRANSFORMER_WEIGHT
        }

        return {
            "name": model.name,
            "version": model.version,
            "type": type(model).__name__,
            "loaded": model.is_trained,
            "weight": weights.get(name.lower(), 0),
            "weights_path": model._weights_path
        }

    def get_all_models_info(self) -> list[dict[str, Any]]:
        """Get information for all models."""
        return [
            self.get_model_info(name)
            for name in self._models
            if self.get_model_info(name) is not None
        ]

    def get_status(self) -> dict[str, Any]:
        """Get registry status."""
        return {
            "total_models": len(self._models),
            "loaded_models": sum(1 for m in self._models.values() if m.is_trained),
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
            "models": {
                name: {"loaded": model.is_trained, "version": model.version}
                for name, model in self._models.items()
            }
        }


# Singleton instance
_registry: Optional[ModelRegistry] = None


async def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        await _registry.initialize()
    return _registry


async def close_model_registry() -> None:
    """Close and cleanup model registry."""
    global _registry
    if _registry is not None:
        # Cleanup any GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _registry = None
        logger.info("Model registry closed")
