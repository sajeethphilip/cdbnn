import json
import os

def generate_config_json(dataset_name, image_shape, num_classes):
    """Generate the processing JSON file."""
    # Extract the number of channels from image_shape
    in_channels = image_shape[0] if len(image_shape) == 3 else 1  # Handle grayscale images

    config = {
        "dataset": {
            "name": dataset_name,
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "input_size": list(image_shape[1:]),  # Height and width
            "mean": [0.5] * in_channels,  # Mean for each channel
            "std": [0.5] * in_channels,   # Std for each channel
            "train_dir": f"data/{dataset_name}/train_data",
            "test_dir": f"data/{dataset_name}/test_data"
        },
        "model": {
            "encoder_type": "cnn",
            "feature_dims": 128,
            "learning_rate": 0.001,
            "optimizer": {
                "type": "Adam",
                "weight_decay": 0.0001,
                "momentum": 0.9,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08
            },
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": 0.1,
                "patience": 10,
                "min_lr": 1e-06,
                "verbose": True
            },
            "loss_functions": {
                "cross_entropy": {
                    "enabled": True,
                    "weight": 1.0
                }
            }
        },
        "training": {
            "batch_size": 32,
            "epochs": 20,
            "num_workers": 4,
            "checkpoint_dir": f"data/{dataset_name}/checkpoints",
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001
            }
        },
        "augmentation": {
            "enabled": True,
            "random_crop": {
                "enabled": True,
                "padding": 4
            },
            "random_rotation": {
                "enabled": True,
                "degrees": 10
            },
            "horizontal_flip": {
                "enabled": True,
                "probability": 0.5
            },
            "color_jitter": {
                "enabled": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            },
            "normalize": {
                "enabled": True,
                "mean": [0.5] * in_channels,
                "std": [0.5] * in_channels
            }
        },
        "execution_flags": {
            "mode": "train_and_predict",
            "use_gpu": True,
            "mixed_precision": True,
            "distributed_training": False,
            "debug_mode": False,
            "use_previous_model": True,
            "fresh_start": False
        },
        "output": {
            "features_file": f"data/{dataset_name}/{dataset_name}.csv",
            "model_dir": f"data/{dataset_name}/models",
            "visualization_dir": f"data/{dataset_name}/visualizations"
        }
    }

    os.makedirs(f"data/{dataset_name}", exist_ok=True)
    with open(f'data/{dataset_name}/{dataset_name}.json', 'w') as f:
        json.dump(config, f, indent=4)

def generate_output_json(dataset_name, image_shape, num_classes):
    """Generate the output JSON file."""
    output_config = {
        "dataset_name": dataset_name,
        "image_shape": image_shape,
        "num_classes": num_classes,
        "output_format": {
            "csv_columns": [f"feature_{i}" for i in range(128)] + ["target"]
        }
    }

    with open(f'data/{dataset_name}/{dataset_name}_output.json', 'w') as f:
        json.dump(output_config, f, indent=4)
