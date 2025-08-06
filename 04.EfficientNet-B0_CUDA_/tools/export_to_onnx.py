#!/usr/bin/env python3
"""
EfficientNet ONNX Export Tool
Exports EfficientNet models from timm to ONNX format with optimizations
"""

import argparse
import torch
import torch.onnx
import timm
import onnx
from onnx import numpy_helper
import numpy as np
import os

def export_model(model_name, output_path, opset_version=14, fp16=False, dynamic_batch=False):
    """Export a timm model to ONNX format"""
    
    print(f"Loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Dynamic axes configuration
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model exported to: {output_path}")
    
    # Optional FP16 conversion
    if fp16:
        print("Converting to FP16...")
        convert_to_fp16(output_path)

def convert_to_fp16(model_path):
    """Convert ONNX model weights to FP16"""
    model = onnx.load(model_path)
    
    # Convert initializers to FP16
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            float_data = numpy_helper.to_array(initializer)
            fp16_data = float_data.astype(np.float16)
            
            new_initializer = numpy_helper.from_array(fp16_data, initializer.name)
            initializer.CopyFrom(new_initializer)
    
    # Update model to use FP16
    for node in model.graph.node:
        if node.op_type in ['Conv', 'MatMul', 'Gemm']:
            for attr in node.attribute:
                if attr.name == 'dtype':
                    attr.i = onnx.TensorProto.FLOAT16
    
    # Save FP16 model
    fp16_path = model_path.replace('.onnx', '_fp16.onnx')
    onnx.save(model, fp16_path)
    print(f"FP16 model saved to: {fp16_path}")

def main():
    parser = argparse.ArgumentParser(description='Export EfficientNet to ONNX')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                                'efficientnet_b6', 'efficientnet_b7'],
                        help='Model variant to export')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true',
                        help='Convert weights to FP16')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Enable dynamic batch size')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Default output path
    if args.output is None:
        args.output = f'models/{args.model}.onnx'
    
    # Export model
    export_model(
        model_name=args.model,
        output_path=args.output,
        opset_version=args.opset,
        fp16=args.fp16,
        dynamic_batch=args.dynamic_batch
    )
    
    # Verify the exported model
    print("\nVerifying exported model...")
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    # Print model info
    print(f"\nModel info:")
    print(f"  Input shape:  {model.graph.input[0].type.tensor_type.shape}")
    print(f"  Output shape: {model.graph.output[0].type.tensor_type.shape}")
    print(f"  Opset:        {model.opset_import[0].version}")

if __name__ == '__main__':
    main()