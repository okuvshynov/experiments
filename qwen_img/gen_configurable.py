from diffusers import DiffusionPipeline # pyright: ignore[reportPrivateImportUsage]
import torch
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Qwen-Image model")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image",
                        help="Model name or path (default: Qwen/Qwen-Image)")
    
    # Prompt configuration
    parser.add_argument("--prompt", type=str, 
                        default="Two zombies in tracksuits sitting in the rusty old soviet car.",
                        help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default=" ",
                        help="Negative prompt (default: single space)")
    
    # Image configuration
    parser.add_argument("--aspect-ratio", type=str, default="320x240",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "640x480", "320x240", "custom"],
                        help="Aspect ratio preset (default: 320x240)")
    parser.add_argument("--width", type=int, default=None,
                        help="Custom width (used when aspect-ratio is 'custom')")
    parser.add_argument("--height", type=int, default=None,
                        help="Custom height (used when aspect-ratio is 'custom')")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps (default: 50)")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    
    # Output configuration
    parser.add_argument("--output", type=str, default="example.png",
                        help="Output filename or pattern (default: example.png)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (default: current directory)")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="Data type for model (default: auto-select based on device)")
    
    # Advanced options
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt from file")
    parser.add_argument("--config", type=str, default=None,
                        help="Load configuration from JSON file")
    parser.add_argument("--save-config", type=str, default=None,
                        help="Save current configuration to JSON file")
    
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def save_config(args, config_file):
    """Save configuration to JSON file"""
    config = vars(args).copy()
    # Remove file-related args that shouldn't be saved
    config.pop('config', None)
    config.pop('save_config', None)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_file}")

def get_dimensions(args):
    """Get width and height based on aspect ratio or custom dimensions"""
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "640x480": (640, 480),
        "320x240": (320, 240),
    }
    
    if args.aspect_ratio == "custom":
        if args.width is None or args.height is None:
            raise ValueError("Width and height must be specified when using custom aspect ratio")
        return args.width, args.height
    else:
        return aspect_ratios[args.aspect_ratio]

def get_device_and_dtype(args):
    """Determine device and dtype based on arguments and availability"""
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    if args.dtype == "auto":
        if device == "cuda":
            torch_dtype = torch.bfloat16
        elif device == "mps":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_map[args.dtype]
    
    return device, torch_dtype

def main():
    args = parse_args()
    
    # Load config from file if specified
    if args.config:
        config_data = load_config(args.config)
        # Update args with config file values (command line args take precedence)
        parser = argparse.ArgumentParser()
        for key, value in config_data.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Save config if requested
    if args.save_config:
        save_config(args, args.save_config)
        return
    
    # Load prompt from file if specified
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            args.prompt = f.read().strip()
    
    # Get device and dtype
    device, torch_dtype = get_device_and_dtype(args)
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # Load the pipeline
    print(f"Loading model: {args.model}")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    
    # Get dimensions
    width, height = get_dimensions(args)
    print(f"Generating {args.batch_size} image(s) at {width}x{height}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg_scale,
        generator=generator,
        num_images_per_prompt=args.batch_size
    ).images # type: ignore
    
    # Save images
    if args.batch_size == 1:
        output_path = output_dir / args.output
        images[0].save(output_path)
        print(f"Image saved to {output_path}")
    else:
        # Save multiple images with numbered filenames
        base_name = Path(args.output).stem
        extension = Path(args.output).suffix or ".png"
        for i, image in enumerate(images):
            output_path = output_dir / f"{base_name}_{i:03d}{extension}"
            image.save(output_path)
            print(f"Image {i+1} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Qwen-Image model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image")
    main()