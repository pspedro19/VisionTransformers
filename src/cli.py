"""
Command Line Interface for ViT-GIF Highlight.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import yaml

from .core.pipeline import InMemoryPipeline


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def process_single_video(args):
    """Process a single video file."""
    try:
        pipeline = InMemoryPipeline(args.config)
        
        # Override config with CLI arguments
        override_config = {}
        
        if args.fps:
            override_config.setdefault('gif', {})['fps'] = args.fps
            
        if args.max_frames:
            override_config.setdefault('gif', {})['max_frames'] = args.max_frames
            
        if args.overlay_style:
            override_config.setdefault('gif', {})['overlay_style'] = args.overlay_style
            
        if args.overlay_intensity is not None:
            override_config.setdefault('gif', {})['overlay_intensity'] = args.overlay_intensity
            
        if args.model:
            override_config.setdefault('model', {})['name'] = args.model
            
        if args.device:
            override_config.setdefault('model', {})['device'] = args.device
        
        # Process video
        result = pipeline.process_video(
            args.input,
            args.output,
            override_config if override_config else None
        )
        
        if result['success']:
            print(f"✅ Successfully created GIF: {result['output_gif']}")
            print(f"📊 Processing time: {result['processing_time']:.2f}s")
            print(f"🎬 Frames: {result['selected_frames']}/{result['total_frames']} selected")
            print(f"📁 File size: {result['gif_stats']['file_size_mb']:.1f}MB")
            print(f"🗜️ Compression ratio: {result['gif_stats']['compression_ratio']:.1f}x")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def process_batch(args):
    """Process multiple videos in batch."""
    try:
        pipeline = InMemoryPipeline(args.config)
        
        # Get list of video files
        input_path = Path(args.input)
        if input_path.is_dir():
            # Process all videos in directory
            video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
            video_files = [
                str(f) for f in input_path.iterdir() 
                if f.suffix.lower() in video_extensions
            ]
        else:
            # Single file
            video_files = [str(input_path)]
        
        if not video_files:
            print("❌ No video files found")
            sys.exit(1)
            
        print(f"🎬 Found {len(video_files)} video(s) to process")
        
        # Process batch
        results = pipeline.process_batch(video_files, args.output)
        
        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        print(f"\n📊 Batch processing completed:")
        print(f"   ✅ Successful: {successful}/{len(results)}")
        print(f"   ⏱️ Total time: {total_time:.2f}s")
        print(f"   📁 Output directory: {args.output}")
        
        # List failed files
        failed_files = [r for r in results if not r.get('success', False)]
        if failed_files:
            print(f"\n❌ Failed files:")
            for result in failed_files:
                print(f"   - {result.get('input_video', 'Unknown')}: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def preview_video(args):
    """Preview video information without processing."""
    try:
        pipeline = InMemoryPipeline(args.config)
        preview = pipeline.get_video_preview(args.input)
        
        if 'error' in preview:
            print(f"❌ Error reading video: {preview['error']}")
            sys.exit(1)
            
        info = preview['video_info']
        limits = preview['within_limits']
        
        print(f"🎬 Video Information: {args.input}")
        print(f"   📐 Resolution: {info['width']}x{info['height']}")
        print(f"   ⏱️ Duration: {info['duration']:.1f}s")
        print(f"   🎞️ Frames: {info['total_frames']} @ {info['fps']:.1f} FPS")
        print(f"   📁 File size: {info['file_size'] / (1024*1024):.1f}MB")
        
        print(f"\n🔍 Processing Check:")
        print(f"   Resolution: {'✅' if limits['resolution'] else '❌'}")
        print(f"   Duration: {'✅' if limits['duration'] else '❌'}")
        print(f"   File size: {'✅' if limits['file_size'] else '❌'}")
        
        if preview['can_process']:
            print(f"   🚀 Can process: ✅")
            print(f"   ⏱️ Estimated time: {preview['estimated_processing_time']:.1f}s")
            
            # Show recommended settings
            rec = preview['recommended_settings']
            print(f"\n💡 Recommended settings:")
            print(f"   FPS: {rec['gif']['fps']}")
            print(f"   Max frames: {rec['gif']['max_frames']}")
            print(f"   Model: {rec['processing']['model']}")
        else:
            print(f"   🚀 Can process: ❌")
            print(f"   Check the limits in your configuration file")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def list_models(args):
    """List available models."""
    try:
        from .models.model_factory import ModelFactory
        
        factory = ModelFactory()
        models = factory.list_available_models()
        
        print("🤖 Available Models:")
        for name, description in models.items():
            info = factory.get_model_info(name)
            memory = info['estimated_memory']
            print(f"\n   📦 {name}")
            print(f"      {description}")
            print(f"      Memory: GPU ~{memory['gpu']}, CPU ~{memory['cpu']}")
            print(f"      Input: {info['input_size']} (frames, H, W)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="ViT-GIF Highlight: Generate intelligent GIFs from videos using Vision Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  vitgif process video.mp4 output.gif
  
  # With custom settings
  vitgif process video.mp4 output.gif --fps 8 --model videomae-large
  
  # Batch processing
  vitgif batch /path/to/videos/ /path/to/output/ --config config/mvp2.yaml
  
  # Preview video
  vitgif preview video.mp4
  
  # List available models
  vitgif models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single video')
    process_parser.add_argument('input', help='Input video file')
    process_parser.add_argument('output', help='Output GIF file')
    process_parser.add_argument('--config', default='config/mvp1.yaml', help='Configuration file')
    process_parser.add_argument('--fps', type=int, help='GIF frames per second')
    process_parser.add_argument('--max-frames', type=int, help='Maximum number of frames')
    process_parser.add_argument('--overlay-style', choices=['heatmap', 'highlight', 'glow', 'pulse'], help='Attention overlay style')
    process_parser.add_argument('--overlay-intensity', type=float, help='Overlay intensity (0.0-1.0)')
    process_parser.add_argument('--model', help='Model to use for attention')
    process_parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos')
    batch_parser.add_argument('input', help='Input directory or file list')
    batch_parser.add_argument('output', help='Output directory')
    batch_parser.add_argument('--config', default='config/mvp1.yaml', help='Configuration file')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview video information')
    preview_parser.add_argument('input', help='Input video file')
    preview_parser.add_argument('--config', default='config/mvp1.yaml', help='Configuration file')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    # Global options
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--version', action='version', version='ViT-GIF Highlight 2.0.0')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == 'process':
        process_single_video(args)
    elif args.command == 'batch':
        process_batch(args)
    elif args.command == 'preview':
        preview_video(args)
    elif args.command == 'models':
        list_models(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 