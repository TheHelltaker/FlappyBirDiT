import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path

def load_h5_data(file_path):
    """Load all chunks from H5 file and combine them."""
    frames = []
    inputs = []
    timestamps = []
    
    with h5py.File(file_path, 'r') as f:
        # Get all chunk names and sort them
        chunk_names = [name for name in f.keys() if name.startswith('chunk_')]
        chunk_names.sort()
        
        print(f"Found {len(chunk_names)} chunks in H5 file")
        
        for chunk_name in chunk_names:
            chunk = f[chunk_name]
            
            # Load frames
            chunk_frames = chunk['frames'][:]
            frames.append(chunk_frames)
            print(f"Loaded {chunk_name}: {len(chunk_frames)} frames")
            
            # Load inputs
            chunk_inputs = chunk['inputs'][:]
            inputs.append(chunk_inputs)
            
            # Load timestamps
            chunk_timestamps = chunk['timestamps'][:]
            timestamps.append(chunk_timestamps)
    
    # Combine all chunks
    all_frames = np.concatenate(frames, axis=0)
    all_inputs = np.concatenate(inputs, axis=0)
    all_timestamps = np.concatenate(timestamps, axis=0)
    
    print(f"Total frames: {len(all_frames)}")
    print(f"Frame shape: {all_frames.shape}")
    
    return all_frames, all_inputs, all_timestamps

def convert_frame_for_opencv(frame):
    """Convert frame to format suitable for OpenCV."""
    # Make sure frame is contiguous and uint8
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    
    # If frame has 4 channels (RGBA), convert to 3 channels (RGB)
    if frame.shape[2] == 4:
        # Remove alpha channel or convert RGBA to RGB
        frame = frame[:, :, :3]
    
    # OpenCV uses BGR, but for video writing RGB is usually fine
    # If you need BGR, uncomment the next line:
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame

def create_annotated_video(frames, timestamps, inputs, input_keys, output_path, fps=None):
    """Create annotated video from frames and input data."""
    
    # Calculate FPS from timestamps if not provided
    if fps is None:
        time_diffs = np.diff(timestamps)
        avg_frame_time = np.mean(time_diffs)
        fps = 1.0 / avg_frame_time
        print(f"Calculated FPS from timestamps: {fps:.2f}")
    
    # Get video dimensions from first frame
    first_frame = convert_frame_for_opencv(frames[0])
    height, width = first_frame.shape[:2]
    
    print(f"Creating video: {output_path}")
    print(f"Video dimensions: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError("Could not open video writer")
    
    try:
        for i in range(len(frames)):
            # Convert frame to OpenCV format
            frame = convert_frame_for_opencv(frames[i])
            
            # Create a copy for annotation to avoid modifying original
            annotated_frame = frame.copy()
            
            # Add input annotations
            y_offset = 30
            for key in input_keys:
                is_pressed = inputs[i][key]
                color = (0, 255, 0) if is_pressed else (0, 0, 255)  # Green if pressed, red if not
                text = f"{key}: {'ON' if is_pressed else 'OFF'}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Draw background rectangle
                cv2.rectangle(annotated_frame, 
                            (5, y_offset - text_height - 5), 
                            (15 + text_width, y_offset + baseline + 5), 
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                y_offset += 35
            
            # Add timestamp
            timestamp_text = f"Time: {timestamps[i]:.3f}s"
            cv2.putText(annotated_frame, timestamp_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"Processed {i}/{len(frames)} frames")
    
    except Exception as e:
        print(f"Error during video creation: {e}")
        raise
    finally:
        out.release()
    
    print(f"Video saved to: {output_path}")

def analyze_inputs(inputs, input_keys):
    """Analyze input activity."""
    print("\nInput Activity Analysis:")
    print("-" * 30)
    
    for key in input_keys:
        key_data = [inp[key] for inp in inputs]
        total_frames = len(key_data)
        on_frames = sum(key_data)
        off_frames = total_frames - on_frames
        
        print(f"Key '{key}':")
        print(f"  Total frames: {total_frames}")
        print(f"  ON frames: {on_frames} ({on_frames/total_frames*100:.1f}%)")
        print(f"  OFF frames: {off_frames} ({off_frames/total_frames*100:.1f}%)")
        
        # Find first few ON frames
        on_frame_indices = [i for i, val in enumerate(key_data) if val]
        if on_frame_indices:
            print(f"  First few ON frames: {on_frame_indices[:10]}")

def main():
    parser = argparse.ArgumentParser(description='Convert H5 game recording to annotated video')
    parser.add_argument('input', help='Input H5 file path')
    parser.add_argument('--output', '-o', help='Output video file path (default: input_filename.mp4 in same directory)')
    parser.add_argument('--fps', type=float, help='Output video FPS (calculated from timestamps if not provided)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_filename = input_path.stem + '.mp4'
        args.output = input_path.parent / output_filename
        print(f"Output will be saved to: {args.output}")
    
    # Load data
    frames, inputs, timestamps = load_h5_data(args.input)
    
    # Get input keys
    input_keys = list(inputs[0].dtype.names)
    print(f"Input keys: {input_keys}")
    
    # Analyze inputs
    analyze_inputs(inputs, input_keys)
    
    # Create video
    try:
        create_annotated_video(frames, timestamps, inputs, input_keys, 
                             args.output, args.fps)
        print("Video creation completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())