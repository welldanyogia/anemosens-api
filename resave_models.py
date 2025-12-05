#!/usr/bin/env python3
"""
resave_models.py - Script to re-save legacy Keras models in compatible format

This script handles compatibility issues when loading old Keras/TensorFlow models
and re-saves them in a format compatible with newer TensorFlow versions.

Usage:
    python resave_models.py

    # Or specify custom paths:
    python resave_models.py --v1 path/to/v1.h5 --v2 path/to/v2.h5

Author: AneMoSense Team
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_tensorflow():
    """Check TensorFlow installation and version."""
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow version: {tf.__version__}")
        logger.info(f"✅ Keras version: {tf.keras.__version__}")
        return tf
    except ImportError:
        logger.error("❌ TensorFlow not installed!")
        logger.error("Install with: pip install tensorflow==2.12.0")
        sys.exit(1)


def load_model_robust(model_path, tf):
    """
    Attempt to load model with multiple strategies.

    Args:
        model_path: Path to the model file
        tf: TensorFlow module

    Returns:
        Loaded model or None
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading model: {model_path}")
    logger.info(f"{'='*60}")

    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return None

    # Strategy 1: Direct load (works if versions match)
    logger.info("📥 Strategy 1: Direct load...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("✅ Model loaded successfully with direct load!")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Direct load failed: {e}")

    # Strategy 2: Load with custom objects
    logger.info("📥 Strategy 2: Load with custom objects...")
    try:
        # Create custom object scope for DTypePolicy
        from tensorflow.python.keras.mixed_precision import policy

        custom_objects = {
            'DTypePolicy': policy.Policy,
        }

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        logger.info("✅ Model loaded successfully with custom objects!")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Custom objects load failed: {e}")

    # Strategy 3: Load weights only
    logger.info("📥 Strategy 3: Load weights only (requires model architecture)...")
    try:
        import h5py
        import json

        # Read model config from h5 file
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config:
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                config = json.loads(model_config)

                # Try to reconstruct model from config
                model = tf.keras.models.model_from_json(json.dumps(config))

                # Load weights
                model.load_weights(model_path)
                logger.info("✅ Model loaded successfully from weights!")
                return model
    except Exception as e:
        logger.warning(f"⚠️  Weights-only load failed: {e}")

    logger.error("❌ All loading strategies failed!")
    return None


def save_model_multiple_formats(model, original_path, output_dir="models_fixed"):
    """
    Save model in multiple formats for maximum compatibility.

    Args:
        model: Loaded Keras model
        original_path: Original model file path
        output_dir: Output directory for saved models
    """
    if model is None:
        logger.error("❌ Cannot save - model is None")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get base filename without extension
    base_name = Path(original_path).stem

    logger.info(f"\n{'='*60}")
    logger.info(f"Saving model: {base_name}")
    logger.info(f"{'='*60}")

    success = False

    # Format 1: HDF5 format (compatible)
    try:
        h5_path = output_path / f"{base_name}_fixed.h5"
        logger.info(f"💾 Saving as HDF5: {h5_path}")
        model.save(str(h5_path), save_format='h5')
        logger.info(f"✅ HDF5 saved successfully!")
        logger.info(f"   Size: {h5_path.stat().st_size / 1024 / 1024:.2f} MB")
        success = True
    except Exception as e:
        logger.error(f"❌ HDF5 save failed: {e}")

    # Format 2: Keras format (Keras 3 native)
    try:
        keras_path = output_path / f"{base_name}_fixed.keras"
        logger.info(f"💾 Saving as Keras: {keras_path}")
        model.save(str(keras_path), save_format='keras')
        logger.info(f"✅ Keras format saved successfully!")
        logger.info(f"   Size: {keras_path.stat().st_size / 1024 / 1024:.2f} MB")
        success = True
    except Exception as e:
        logger.warning(f"⚠️  Keras format save failed: {e}")

    # Format 3: SavedModel format (most reliable for TF)
    try:
        savedmodel_path = output_path / f"{base_name}_savedmodel"
        logger.info(f"💾 Saving as SavedModel: {savedmodel_path}")
        model.save(str(savedmodel_path), save_format='tf')
        logger.info(f"✅ SavedModel saved successfully!")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in savedmodel_path.rglob('*') if f.is_file())
        logger.info(f"   Size: {total_size / 1024 / 1024:.2f} MB")
        success = True
    except Exception as e:
        logger.warning(f"⚠️  SavedModel save failed: {e}")

    return success


def test_load_saved_model(model_path, tf):
    """Test loading the newly saved model."""
    logger.info(f"\n🧪 Testing load of saved model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("✅ Saved model loads successfully!")

        # Print model summary
        logger.info("\n📋 Model Summary:")
        model.summary()

        return True
    except Exception as e:
        logger.error(f"❌ Failed to load saved model: {e}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Re-save legacy Keras models in compatible format'
    )
    parser.add_argument(
        '--v1',
        default='anemia_model_v1.h5',
        help='Path to Model V1 (default: anemia_model_v1.h5)'
    )
    parser.add_argument(
        '--v2',
        default='model_anemia_v2.h5',
        help='Path to Model V2 (default: model_anemia_v2.h5)'
    )
    parser.add_argument(
        '--output-dir',
        default='models_fixed',
        help='Output directory for fixed models (default: models_fixed)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test loading the saved models after conversion'
    )

    args = parser.parse_args()

    # Print header
    logger.info("="*60)
    logger.info("  AneMoSense Model Re-Saver")
    logger.info("  Converting legacy models to compatible format")
    logger.info("="*60)

    # Check TensorFlow
    tf = check_tensorflow()

    # Process Model V1
    logger.info("\n" + "="*60)
    logger.info("PROCESSING MODEL V1 (Lightweight)")
    logger.info("="*60)

    model_v1 = load_model_robust(args.v1, tf)
    if model_v1:
        save_model_multiple_formats(model_v1, args.v1, args.output_dir)

        # Test if requested
        if args.test:
            test_path = Path(args.output_dir) / f"{Path(args.v1).stem}_fixed.h5"
            if test_path.exists():
                test_load_saved_model(str(test_path), tf)

    # Process Model V2
    logger.info("\n" + "="*60)
    logger.info("PROCESSING MODEL V2 (Accurate)")
    logger.info("="*60)

    model_v2 = load_model_robust(args.v2, tf)
    if model_v2:
        save_model_multiple_formats(model_v2, args.v2, args.output_dir)

        # Test if requested
        if args.test:
            test_path = Path(args.output_dir) / f"{Path(args.v2).stem}_fixed.h5"
            if test_path.exists():
                test_load_saved_model(str(test_path), tf)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    output_path = Path(args.output_dir)
    if output_path.exists():
        saved_files = list(output_path.glob('*'))
        logger.info(f"✅ Output directory: {output_path.absolute()}")
        logger.info(f"✅ Files created: {len(saved_files)}")

        for file in sorted(saved_files):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                logger.info(f"   - {file.name} ({size_mb:.2f} MB)")
            elif file.is_dir():
                total_size = sum(f.stat().st_size for f in file.rglob('*') if f.is_file())
                size_mb = total_size / 1024 / 1024
                logger.info(f"   - {file.name}/ ({size_mb:.2f} MB)")

    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Update app.py to use the new model files:")
    logger.info("   MODEL_V1_PATH = 'models_fixed/anemia_model_v1_fixed.h5'")
    logger.info("   MODEL_V2_PATH = 'models_fixed/model_anemia_v2_fixed.h5'")
    logger.info("")
    logger.info("2. Or use SavedModel format (more reliable):")
    logger.info("   MODEL_V1_PATH = 'models_fixed/anemia_model_v1_savedmodel'")
    logger.info("   MODEL_V2_PATH = 'models_fixed/model_anemia_v2_savedmodel'")
    logger.info("")
    logger.info("3. Test the application:")
    logger.info("   python app.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
