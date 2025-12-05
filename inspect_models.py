#!/usr/bin/env python3
"""
inspect_models.py - Model Audit Script for AneMoSense

Script untuk mengaudit struktur model ML sebelum implementasi Dual-Model Prediction.
Menampilkan informasi detail tentang input/output shape, layer structure, dan metadata.

Usage:
    python inspect_models.py
    
Requirements:
    - tensorflow >= 2.x
    - h5py (biasanya sudah include dengan tensorflow)
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
except ImportError as e:
    print(f"❌ Error: Missing dependency - {e}")
    print("   Install with: pip install tensorflow numpy")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_FILES = [
    'model_anemia.h5',      # Current production model
    'anemia_model_v1.h5',   # Version 1
    'model_anemia_v2.h5',   # Version 2
]

# Handle both script execution and Colab/Jupyter environment
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    # Running in Colab/Jupyter - use current working directory
    SCRIPT_DIR = Path.cwd()


def print_separator(char='═', length=80):
    """Print a separator line."""
    print(char * length)


def print_header(title):
    """Print a section header."""
    print()
    print_separator('═')
    print(f"  {title}")
    print_separator('═')


def inspect_model(model_path: str) -> dict:
    """
    Load and inspect a Keras model.
    
    Returns a dictionary with model information.
    """
    result = {
        'path': model_path,
        'exists': False,
        'error': None,
        'inputs': [],
        'outputs': [],
        'total_params': 0,
        'trainable_params': 0,
        'layers_count': 0,
        'layer_types': [],
    }
    
    full_path = SCRIPT_DIR / model_path
    
    if not full_path.exists():
        result['error'] = f"File not found: {full_path}"
        return result
    
    result['exists'] = True
    result['file_size_mb'] = full_path.stat().st_size / (1024 * 1024)
    
    try:
        # Load model without compiling to avoid optimizer state issues
        model = keras.models.load_model(str(full_path), compile=False)
        
        # Basic info
        result['layers_count'] = len(model.layers)
        result['total_params'] = model.count_params()
        
        # Count trainable params
        trainable_count = sum([
            tf.reduce_prod(var.shape).numpy() 
            for var in model.trainable_variables
        ])
        result['trainable_params'] = int(trainable_count)
        
        # Get unique layer types
        layer_types = set()
        for layer in model.layers:
            layer_types.add(type(layer).__name__)
        result['layer_types'] = sorted(list(layer_types))
        
        # ─── Input Analysis ───────────────────────────────────────────────────
        # Handle both single and multiple inputs
        if hasattr(model, 'inputs') and model.inputs:
            for i, inp in enumerate(model.inputs):
                input_info = {
                    'index': i,
                    'name': inp.name if hasattr(inp, 'name') else f'input_{i}',
                    'shape': tuple(inp.shape.as_list()) if hasattr(inp.shape, 'as_list') else str(inp.shape),
                    'dtype': str(inp.dtype.name) if hasattr(inp, 'dtype') else 'unknown',
                }
                result['inputs'].append(input_info)
        elif hasattr(model, 'input_shape'):
            result['inputs'].append({
                'index': 0,
                'name': 'input',
                'shape': model.input_shape,
                'dtype': 'float32',  # Default assumption
            })
        
        # ─── Output Analysis ──────────────────────────────────────────────────
        if hasattr(model, 'outputs') and model.outputs:
            for i, out in enumerate(model.outputs):
                output_info = {
                    'index': i,
                    'name': out.name if hasattr(out, 'name') else f'output_{i}',
                    'shape': tuple(out.shape.as_list()) if hasattr(out.shape, 'as_list') else str(out.shape),
                    'dtype': str(out.dtype.name) if hasattr(out, 'dtype') else 'unknown',
                }
                result['outputs'].append(output_info)
        elif hasattr(model, 'output_shape'):
            result['outputs'].append({
                'index': 0,
                'name': 'output',
                'shape': model.output_shape,
                'dtype': 'float32',
            })
        
        # ─── First and Last Layer Details ─────────────────────────────────────
        if model.layers:
            first_layer = model.layers[0]
            last_layer = model.layers[-1]
            
            result['first_layer'] = {
                'name': first_layer.name,
                'type': type(first_layer).__name__,
                'config': first_layer.get_config() if hasattr(first_layer, 'get_config') else {},
            }
            
            result['last_layer'] = {
                'name': last_layer.name,
                'type': type(last_layer).__name__,
                'config': last_layer.get_config() if hasattr(last_layer, 'get_config') else {},
            }
            
            # Check for activation in last layer
            if hasattr(last_layer, 'activation'):
                result['last_layer']['activation'] = last_layer.activation.__name__
        
        # ─── Model Summary (capture as string) ────────────────────────────────
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        result['summary'] = '\n'.join(string_list)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def print_model_info(info: dict):
    """Pretty print model information."""
    
    model_name = Path(info['path']).stem
    print_header(f"MODEL: {info['path']}")
    
    if not info['exists']:
        print(f"  ❌ {info['error']}")
        return
    
    if info['error']:
        print(f"  ⚠️  Error loading model: {info['error']}")
        return
    
    # Basic Stats
    print(f"\n📊 BASIC INFORMATION:")
    print(f"   File Size     : {info['file_size_mb']:.2f} MB")
    print(f"   Total Layers  : {info['layers_count']}")
    print(f"   Total Params  : {info['total_params']:,}")
    print(f"   Trainable     : {info['trainable_params']:,}")
    print(f"   Layer Types   : {', '.join(info['layer_types'])}")
    
    # Input Information
    print(f"\n📥 INPUT SPECIFICATION:")
    for inp in info['inputs']:
        print(f"   [{inp['index']}] {inp['name']}")
        print(f"       Shape : {inp['shape']}")
        print(f"       Dtype : {inp['dtype']}")
    
    # Output Information  
    print(f"\n📤 OUTPUT SPECIFICATION:")
    for out in info['outputs']:
        print(f"   [{out['index']}] {out['name']}")
        print(f"       Shape : {out['shape']}")
        print(f"       Dtype : {out['dtype']}")
    
    # First/Last Layer
    if 'first_layer' in info:
        print(f"\n🔷 FIRST LAYER:")
        print(f"   Name : {info['first_layer']['name']}")
        print(f"   Type : {info['first_layer']['type']}")
    
    if 'last_layer' in info:
        print(f"\n🔶 LAST LAYER:")
        print(f"   Name : {info['last_layer']['name']}")
        print(f"   Type : {info['last_layer']['type']}")
        if 'activation' in info['last_layer']:
            print(f"   Activation : {info['last_layer']['activation']}")
    
    # Model Summary (optional - can be verbose)
    print(f"\n📋 MODEL SUMMARY:")
    print_separator('─', 60)
    if 'summary' in info:
        # Print truncated summary (first 30 lines)
        lines = info['summary'].split('\n')
        for line in lines[:35]:
            print(f"   {line}")
        if len(lines) > 35:
            print(f"   ... ({len(lines) - 35} more lines)")
    print_separator('─', 60)


def generate_comparison_table(all_info: list):
    """Generate a comparison table for all models."""
    
    print_header("📊 MODEL COMPARISON TABLE")
    
    # Filter only successfully loaded models
    valid_models = [m for m in all_info if m['exists'] and not m['error']]
    
    if not valid_models:
        print("  No valid models to compare.")
        return
    
    print(f"\n{'Model':<25} {'Size (MB)':<12} {'Params':<15} {'Inputs':<20} {'Output':<15}")
    print_separator('─', 90)
    
    for m in valid_models:
        name = Path(m['path']).stem[:24]
        size = f"{m['file_size_mb']:.2f}"
        params = f"{m['total_params']:,}"
        
        # Format inputs
        if m['inputs']:
            inp_shapes = [str(i['shape']) for i in m['inputs']]
            inputs = ', '.join(inp_shapes)[:19]
        else:
            inputs = 'N/A'
        
        # Format output
        if m['outputs']:
            output = str(m['outputs'][0]['shape'])[:14]
        else:
            output = 'N/A'
        
        print(f"{name:<25} {size:<12} {params:<15} {inputs:<20} {output:<15}")
    
    print_separator('─', 90)


def generate_code_snippet(all_info: list):
    """Generate Python code snippet for loading models."""
    
    print_header("💻 SUGGESTED CODE SNIPPET")
    
    valid_models = [m for m in all_info if m['exists'] and not m['error']]
    
    if not valid_models:
        print("  No valid models found.")
        return
    
    print("\n# ─── Model Loading Code ───────────────────────────────────────")
    print("import tensorflow as tf")
    print()
    print("# Model paths")
    
    for m in valid_models:
        var_name = Path(m['path']).stem.upper().replace('-', '_').replace('.', '_')
        print(f"{var_name}_PATH = '{m['path']}'")
    
    print()
    print("# Load models")
    for m in valid_models:
        var_name = Path(m['path']).stem.replace('-', '_').replace('.', '_')
        print(f"{var_name} = tf.keras.models.load_model('{m['path']}', compile=False)")
    
    print()
    print("# ─── Input Preprocessing ──────────────────────────────────────")
    
    # Show input requirements for first valid model
    if valid_models and valid_models[0]['inputs']:
        m = valid_models[0]
        print(f"# Model: {m['path']}")
        print(f"# Expected inputs: {len(m['inputs'])}")
        for inp in m['inputs']:
            print(f"#   - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")


def main():
    """Main entry point."""
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           AneMoSense - Model Inspection Tool v1.0                            ║")
    print("║           Audit ML Models Before Dual-Model Implementation                   ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n🔍 Scanning for models in: {SCRIPT_DIR}")
    print(f"   TensorFlow version: {tf.__version__}")
    
    # Inspect all models
    all_info = []
    for model_file in MODEL_FILES:
        info = inspect_model(model_file)
        all_info.append(info)
        print_model_info(info)
    
    # Generate comparison table
    generate_comparison_table(all_info)
    
    # Generate code snippet
    generate_code_snippet(all_info)
    
    print()
    print_separator('═')
    print("  ✅ Model inspection complete!")
    print("  📝 Copy the results above and share with the development team.")
    print_separator('═')
    print()


if __name__ == '__main__':
    main()

