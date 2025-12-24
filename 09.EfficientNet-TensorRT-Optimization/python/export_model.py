#!/usr/bin/env python3
"""
EfficientNet ONNX Export Script

Exports EfficientNet models (B0-B7) to ONNX format for TensorRT optimization.
Supports both timm and torchvision model sources.
"""

import argparse
import os
import sys

import torch
import torch.onnx


def get_model(model_name: str, pretrained: bool = True):
    """Load EfficientNet model from timm or torchvision."""

    # Try timm first (more complete EfficientNet support)
    try:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained)
        print(f"Loaded {model_name} from timm")
        return model, timm.data.resolve_data_config(model=model)
    except ImportError:
        pass
    except Exception as e:
        print(f"timm failed: {e}")

    # Fall back to torchvision
    try:
        import torchvision.models as models

        model_map = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
        }

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        weights = 'IMAGENET1K_V1' if pretrained else None
        model = model_map[model_name](weights=weights)
        print(f"Loaded {model_name} from torchvision")

        # Default config for torchvision models
        config = {
            'input_size': (3, 224, 224),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
        }
        return model, config

    except Exception as e:
        print(f"torchvision failed: {e}")
        raise RuntimeError(f"Could not load model {model_name}")


def export_to_onnx(
    model_name: str,
    output_path: str,
    batch_size: int = 1,
    input_size: int = 224,
    opset_version: int = 17,
    dynamic_batch: bool = False,
    simplify: bool = True,
    fp16: bool = False
):
    """Export model to ONNX format."""

    print(f"\n{'='*50}")
    print(f"Exporting {model_name} to ONNX")
    print(f"{'='*50}")

    # Load model
    model, config = get_model(model_name, pretrained=True)
    model.eval()

    # Use config input size if available
    if 'input_size' in config:
        input_size = config['input_size'][-1]

    print(f"Input size: {input_size}x{input_size}")
    print(f"Batch size: {batch_size}")
    print(f"ONNX opset: {opset_version}")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # Dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print("Dynamic batch: enabled")

    # Export
    print(f"\nExporting to: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    print(f"Export successful!")

    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: passed")

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print("Simplification successful!")
            else:
                print("Simplification failed, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")

    # Convert to FP16 if requested
    if fp16:
        try:
            from onnxmltools.utils import float16_converter
            print("\nConverting to FP16...")
            fp16_model = float16_converter.convert_float_to_float16(onnx_model)
            fp16_path = output_path.replace('.onnx', '_fp16.onnx')
            onnx.save(fp16_model, fp16_path)
            print(f"FP16 model saved to: {fp16_path}")
        except ImportError:
            print("onnxmltools not installed, skipping FP16 conversion")

    # Print model info
    print(f"\n{'='*50}")
    print("Model Info")
    print(f"{'='*50}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")

    # Print preprocessing info
    print(f"\n{'='*50}")
    print("Preprocessing Config (save this for inference)")
    print(f"{'='*50}")
    print(f"Input size: {input_size}")
    print(f"Mean: {config.get('mean', (0.485, 0.456, 0.406))}")
    print(f"Std: {config.get('std', (0.229, 0.224, 0.225))}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Export EfficientNet to ONNX for TensorRT'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='efficientnet_b0',
        help='Model name (efficientnet_b0 to b7)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output ONNX file path'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--input-size', '-s',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--opset', '-p',
        type=int,
        default=17,
        help='ONNX opset version (default: 17)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX simplification'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Also export FP16 version'
    )

    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f'{args.model}.onnx')

    export_to_onnx(
        model_name=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
        input_size=args.input_size,
        opset_version=args.opset,
        dynamic_batch=args.dynamic,
        simplify=not args.no_simplify,
        fp16=args.fp16
    )


if __name__ == '__main__':
    main()
