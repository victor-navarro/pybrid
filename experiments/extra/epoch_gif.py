from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np


def create_frames(N):
    frames = []
    for i in range(1, N + 1):
        # Create an image with white background
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Load a font
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text size and position
        text = str(i)
        text_width, text_height = d.textsize(text, font=font)
        position = ((img.width - text_width) // 2, (img.height - text_height) // 2)

        # Draw the text on the image
        d.text(position, text, fill=(0, 0, 0), font=font)

        # Append the frame
        frames.append(img)
    return frames


def save_gif(frames, filename):
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=10, loop=0
    )


def save_webm(frames, filename, duration):
    fps = len(frames) / duration
    with imageio.get_writer(filename, fps=fps, codec="libvpx-vp9") as writer:
        for frame in frames:
            frame_array = np.array(frame)
            writer.append_data(frame_array)


def main(N):
    frames = create_frames(N)
    save_gif(frames, f"experiments/extra/{N}.gif")
    save_webm(frames, f"experiments/extra/{N}.webm", N / 10)


if __name__ == "__main__":
    main(100)
    main(50)
