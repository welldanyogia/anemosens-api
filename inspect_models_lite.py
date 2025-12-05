#!/usr/bin/env python3
"""
inspect_models_lite.py - Lightweight Model Audit (No TensorFlow Required)

Script untuk mengaudit struktur model H5 tanpa memerlukan TensorFlow.
Menggunakan h5py untuk membaca struktur file HDF5 secara langsung.

Usage:
    pip install h5py numpy
    python inspect_models_lite.py
"""

import os
import sys
import json
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"❌ Error: Missing dependency - {e}")
    print("   Install with: pip install h5py numpy")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_FILES = [
    'model_anemia.h5',
    'anemia_model_v1.h5', 
    'model_anemia_v2.h5',
]

# Handle both script execution and Colab/Jupyter environment
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    # Running in Colab/Jupyter - use current working directory
    SCRIPT_DIR = Path.cwd()


def print_separator(char='═', length=80):
    print(char * length)


def print_header(title):
    print()
    print_separator('═')
    print(f"  {title}")
    print_separator('═')


def get_h5_structure(group, prefix=''):
    """Recursively get structure of H5 file."""
    structure = {}
    
    for key in group.keys():
        item = group[key]
        full_path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            structure[key] = {
                'type': 'group',
                'children': get_h5_structure(item, full_path)
            }
        elif isinstance(item, h5py.Dataset):
            structure[key] = {
                'type': 'dataset',
                'shape': item.shape,
                'dtype': str(item.dtype),
                'size': item.size
            }
    
    return structure


def extract_model_config(h5file):
    """Extract Keras model configuration from H5 file."""
    config = {}
    
    # Try to get model config from attributes
    if 'model_config' in h5file.attrs:
        try:
            model_config_raw = h5file.attrs['model_config']
            if isinstance(model_config_raw, bytes):
                model_config_raw = model_config_raw.decode('utf-8')
            config['model_config'] = json.loads(model_config_raw)
        except Exception as e:
            config['model_config_error'] = str(e)
    
    # Try keras_version
    if 'keras_version' in h5file.attrs:
        config['keras_version'] = h5file.attrs['keras_version']
        if isinstance(config['keras_version'], bytes):
            config['keras_version'] = config['keras_version'].decode('utf-8')
    
    # Try backend
    if 'backend' in h5file.attrs:
        config['backend'] = h5file.attrs['backend']
        if isinstance(config['backend'], bytes):
            config['backend'] = config['backend'].decode('utf-8')
    
    return config


def extract_layer_info(model_config):
    """Extract layer information from model config."""
    layers = []
    
    if not model_config:
        return layers
    
    config = model_config.get('config', {})
    
    # Handle Sequential models
    if 'layers' in config:
        for layer in config['layers']:
            layer_info = {
                'class_name': layer.get('class_name', 'Unknown'),
                'name': layer.get('config', {}).get('name', 'unnamed'),
                'config': layer.get('config', {})
            }
            layers.append(layer_info)
    
    # Handle Functional models
    elif 'input_layers' in config:
        # Get input layers
        for inp in config.get('input_layers', []):
            layers.append({
                'class_name': 'InputLayer',
                'name': inp[0] if isinstance(inp, list) else str(inp),
                'is_input': True
            })
        
        # Get all layers
        for layer in config.get('layers', []):
            layer_info = {
                'class_name': layer.get('class_name', 'Unknown'),
                'name': layer.get('name', 'unnamed'),
                'config': layer.get('config', {}),
                'inbound_nodes': layer.get('inbound_nodes', [])
            }
            layers.append(layer_info)
    
    return layers


def extract_input_output_shapes(model_config, layers):
    """Try to extract input and output shapes."""
    inputs = []
    outputs = []
    
    if not model_config:
        return inputs, outputs
    
    config = model_config.get('config', {})
    
    # Get input layers info
    for layer in layers:
        if layer.get('class_name') == 'InputLayer' or layer.get('is_input'):
            layer_config = layer.get('config', {})
            batch_input_shape = layer_config.get('batch_input_shape') or layer_config.get('batch_shape')
            
            if batch_input_shape:
                inputs.append({
                    'name': layer.get('name', 'input'),
                    'shape': tuple(batch_input_shape),
                    'dtype': layer_config.get('dtype', 'float32')
                })
    
    # Get output layers
    output_layers = config.get('output_layers', [])
    for out in output_layers:
        out_name = out[0] if isinstance(out, list) else str(out)
        outputs.append({'name': out_name})
    
    # If no explicit output, use last layer
    if not outputs and layers:
        last_layer = layers[-1]
        outputs.append({
            'name': last_layer.get('name', 'output'),
            'class_name': last_layer.get('class_name')
        })
    
    return inputs, outputs


def inspect_model(model_path: str) -> dict:
    """Inspect a Keras H5 model file."""
    result = {
        'path': model_path,
        'exists': False,
        'error': None,
    }
    
    full_path = SCRIPT_DIR / model_path
    
    if not full_path.exists():
        result['error'] = f"File not found: {full_path}"
        return result
    
    result['exists'] = True
    result['file_size_mb'] = full_path.stat().st_size / (1024 * 1024)
    
    try:
        with h5py.File(full_path, 'r') as h5file:
            # Get basic structure
            result['h5_structure'] = list(h5file.keys())
            
            # Get attributes
            result['attributes'] = list(h5file.attrs.keys())
            
            # Extract Keras config
            config_data = extract_model_config(h5file)
            result['keras_version'] = config_data.get('keras_version', 'Unknown')
            result['backend'] = config_data.get('backend', 'Unknown')
            
            model_config = config_data.get('model_config', {})
            result['model_class'] = model_config.get('class_name', 'Unknown')
            
            # Extract layers
            layers = extract_layer_info(model_config)
            result['layers_count'] = len(layers)
            result['layers'] = layers
            
            # Get unique layer types
            layer_types = set()
            for layer in layers:
                layer_types.add(layer.get('class_name', 'Unknown'))
            result['layer_types'] = sorted(list(layer_types))
            
            # Extract input/output shapes
            inputs, outputs = extract_input_output_shapes(model_config, layers)
            result['inputs'] = inputs
            result['outputs'] = outputs
            
            # Count weights
            if 'model_weights' in h5file:
                weights_group = h5file['model_weights']
                result['weight_layers'] = list(weights_group.keys())
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def print_model_info(info: dict):
    """Pretty print model information."""
    
    print_header(f"MODEL: {info['path']}")
    
    if not info['exists']:
        print(f"  ❌ {info['error']}")
        return
    
    if info.get('error'):
        print(f"  ⚠️  Error: {info['error']}")
        return
    
    # Basic Stats
    print(f"\n📊 BASIC INFORMATION:")
    print(f"   File Size      : {info['file_size_mb']:.2f} MB")
    print(f"   Keras Version  : {info.get('keras_version', 'Unknown')}")
    print(f"   Backend        : {info.get('backend', 'Unknown')}")
    print(f"   Model Class    : {info.get('model_class', 'Unknown')}")
    print(f"   Total Layers   : {info.get('layers_count', 0)}")
    print(f"   H5 Structure   : {info.get('h5_structure', [])}")
    
    # Layer Types
    print(f"\n🧱 LAYER TYPES:")
    for lt in info.get('layer_types', []):
        print(f"   • {lt}")
    
    # Input Information
    print(f"\n📥 INPUT SPECIFICATION:")
    inputs = info.get('inputs', [])
    if inputs:
        for i, inp in enumerate(inputs):
            print(f"   [{i}] {inp.get('name', 'input')}")
            print(f"       Shape : {inp.get('shape', 'Unknown')}")
            print(f"       Dtype : {inp.get('dtype', 'float32')}")
    else:
        print("   (Could not extract input info - check layers below)")
    
    # Output Information  
    print(f"\n📤 OUTPUT SPECIFICATION:")
    outputs = info.get('outputs', [])
    if outputs:
        for i, out in enumerate(outputs):
            print(f"   [{i}] {out.get('name', 'output')}")
            if 'class_name' in out:
                print(f"       Type : {out['class_name']}")
    else:
        print("   (Could not extract output info)")
    
    # First few layers
    print(f"\n📋 LAYER DETAILS (First 10):")
    print_separator('─', 70)
    layers = info.get('layers', [])
    for i, layer in enumerate(layers[:10]):
        layer_config = layer.get('config', {})
        shape_info = ''
        
        # Try to get shape info
        if 'batch_input_shape' in layer_config:
            shape_info = f" shape={layer_config['batch_input_shape']}"
        elif 'units' in layer_config:
            shape_info = f" units={layer_config['units']}"
        elif 'filters' in layer_config:
            shape_info = f" filters={layer_config['filters']}"
        
        print(f"   {i:2d}. {layer.get('class_name', '?'):<20} name='{layer.get('name', '?')[:20]}'{shape_info}")
    
    if len(layers) > 10:
        print(f"   ... and {len(layers) - 10} more layers")
    print_separator('─', 70)


def print_comparison_table(all_info: list):
    """Print comparison table."""
    print_header("📊 MODEL COMPARISON TABLE")
    
    valid = [m for m in all_info if m['exists'] and not m.get('error')]
    
    if not valid:
        print("  No valid models to compare.")
        return
    
    print(f"\n{'Model':<25} {'Size (MB)':<12} {'Layers':<10} {'Input Shape':<25}")
    print_separator('─', 80)
    
    for m in valid:
        name = Path(m['path']).stem[:24]
        size = f"{m['file_size_mb']:.2f}"
        layers = str(m.get('layers_count', '?'))
        
        inputs = m.get('inputs', [])
        if inputs and 'shape' in inputs[0]:
            inp_shape = str(inputs[0]['shape'])[:24]
        else:
            inp_shape = 'Check layers above'
        
        print(f"{name:<25} {size:<12} {layers:<10} {inp_shape:<25}")
    
    print_separator('─', 80)


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           AneMoSense - Model Inspection Tool (Lite) v1.0                     ║")
    print("║           No TensorFlow Required - Uses h5py Only                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n🔍 Scanning for models in: {SCRIPT_DIR}")
    print(f"   h5py version: {h5py.__version__}")
    
    all_info = []
    for model_file in MODEL_FILES:
        info = inspect_model(model_file)
        all_info.append(info)
        print_model_info(info)
    
    print_comparison_table(all_info)
    
    print()
    print_separator('═')
    print("  ✅ Model inspection complete!")
    print("  📝 Share these results for Dual-Model implementation planning.")
    print_separator('═')
    print()


if __name__ == '__main__':
    main()

