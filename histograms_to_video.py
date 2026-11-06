"""Stitch the histogram images in ./histograms/ into a video.

Requires: pip install imageio imageio-ffmpeg
Falls back to an animated GIF if the ffmpeg backend isn't available.
"""

import re
from pathlib import Path

import imageio.v2 as imageio

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
INPUT_DIR = Path("histograms")
OUTPUT_PATH = Path("histograms_video.mp4")
FPS = 5  # frames per second; lower = slower playback
HOLD_LAST_FRAME_SEC = 2  # pause on the final frame so the end state is visible


def frame_number(path: Path) -> int:
    """Pull the frame number out of names like frame_00011.png."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"'{INPUT_DIR}/' not found. Run histograms.py first to generate the images."
    )

image_paths = sorted(INPUT_DIR.glob("*.png"), key=frame_number)
if not image_paths:
    raise FileNotFoundError(f"No .png images found in '{INPUT_DIR}/'.")

print(f"Found {len(image_paths)} images in '{INPUT_DIR}/'.")

hold_frames = max(0, int(HOLD_LAST_FRAME_SEC * FPS))


def write_frames(writer):
    last = None
    for i, path in enumerate(image_paths, start=1):
        last = imageio.imread(path)
        writer.append_data(last)
        print(f"Added {i}/{len(image_paths)}: {path.name}", end="\r")
    for _ in range(hold_frames):
        writer.append_data(last)


try:
    # macro_block_size=1 keeps the original resolution (no resizing warnings);
    # yuv420p makes the MP4 play in most players.
    with imageio.get_writer(
        OUTPUT_PATH,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    ) as writer:
        write_frames(writer)
    print(f"\nSaved video to '{OUTPUT_PATH}'.")

except (ImportError, ValueError, RuntimeError) as err:
    gif_path = OUTPUT_PATH.with_suffix(".gif")
    print(f"\nCould not write MP4 ({err}). Writing a GIF instead...")
    with imageio.get_writer(gif_path, mode="I", duration=1000 / FPS, loop=0) as writer:
        write_frames(writer)
    print(f"\nSaved GIF to '{gif_path}'.")
