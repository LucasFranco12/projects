from PIL import Image, ImageDraw
import numpy as np
import imageio

def create_frame(size, button_color, triangle_color, cursor_image, mouse_pos=None, draw_play=True):
    # Create a new blank transparent image for each frame
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Adjust button size to be smaller
    button_size = min(size) // 3  # Reduced from // 2 to // 3
    button_pos = ((size[0] - button_size) // 2, (size[1] - button_size) // 2)
    
    if draw_play:
        # Draw play button
        draw.ellipse([button_pos, (button_pos[0] + button_size, button_pos[1] + button_size)], fill=button_color)
        triangle_size = button_size // 2
        triangle_pos = (button_pos[0] + button_size // 4, button_pos[1] + button_size // 4)
        draw.polygon([
            triangle_pos,
            (triangle_pos[0], triangle_pos[1] + triangle_size),
            (triangle_pos[0] + triangle_size, triangle_pos[1] + triangle_size // 2)
        ], fill=triangle_color)
    else:
        # Draw pause button
        rect_width = button_size // 5
        spacing = button_size // 10
        # Move the rectangles to the right by adding a horizontal offset
        offset = button_size // 10  # Adjust the offset for the new size
        rect1_pos = (button_pos[0] + spacing + offset, button_pos[1] + spacing)
        rect2_pos = (button_pos[0] + 3 * spacing + rect_width + offset, button_pos[1] + spacing)
        draw.rectangle([rect1_pos, (rect1_pos[0] + rect_width, rect1_pos[1] + button_size - 2 * spacing)], fill=pause_color)
        draw.rectangle([rect2_pos, (rect2_pos[0] + rect_width, rect2_pos[1] + button_size - 2 * spacing)], fill=pause_color)

    # Overlay mouse cursor
    if mouse_pos:
        # Ensure cursor is pasted with transparency mask
        image.paste(cursor_image, mouse_pos, cursor_image)
    
    return np.array(image)

# Load the mouse cursor images
cursor_path = 'mouse_cursor.png'  # Replace with your default mouse cursor image file name
highlighted_cursor_path = 'item_is_highlighted.png'  # Replace with your highlighted cursor image file name
cursor_image = Image.open(cursor_path).convert('RGBA')
highlighted_cursor_image = Image.open(highlighted_cursor_path).convert('RGBA')

# Resize the cursor images to make them smaller
cursor_size = (20, 20)
cursor_image = cursor_image.resize(cursor_size, Image.LANCZOS)
highlighted_cursor_image = highlighted_cursor_image.resize(cursor_size, Image.LANCZOS)

# Create animation frames
frames = []
size = (400, 400)
button_color = (200, 200, 200, 255)
triangle_color = (50, 50, 50, 255)
pause_color = (255, 255, 255, 255)

# Define the button center
button_center = (size[0] // 2, size[1] // 2)
start_pos = (20, 20)  # Start position of the cursor
button_radius = min(size) // 6  # Button radius for collision detection
total_frames = 150  # Total number of frames (increased for pause)

for i in range(total_frames):
    if i < 60:
        # Animate mouse cursor moving diagonally from top-left to center
        x = start_pos[0] + (button_center[0] - start_pos[0]) * (i / 60)
        y = start_pos[1] + (button_center[1] - start_pos[1]) * (i / 60)
        mouse_pos = (int(x - cursor_size[0] // 2), int(y - cursor_size[1] // 2))
        # Check if cursor is within button area
        distance_to_center = np.sqrt((x - button_center[0]) ** 2 + (y - button_center[1]) ** 2)
        is_on_button = distance_to_center <= button_radius
        frames.append(create_frame(size, button_color, triangle_color, highlighted_cursor_image if is_on_button else cursor_image, mouse_pos, draw_play=True))
    elif i == 60:
        # Pause for 15 frames before clicking
        pause_frames = 15
        for _ in range(pause_frames):
            frames.append(create_frame(size, button_color, triangle_color, highlighted_cursor_image, mouse_pos, draw_play=True))
    elif i < 90:
        # Change to clicked cursor and darken button, start click animation
        mouse_pos = (button_center[0] - cursor_size[0] // 2, button_center[1] - cursor_size[1] // 2)
        frames.append(create_frame(size, (150, 150, 150, 255), triangle_color, highlighted_cursor_image, mouse_pos, draw_play=False))
    else:
        # Show pause button and ensure cursor stays highlighted
        mouse_pos = (button_center[0] - cursor_size[0] // 2, button_center[1] - cursor_size[1] // 2)
        frames.append(create_frame(size, button_color, triangle_color, highlighted_cursor_image, mouse_pos, draw_play=False))

# Save as GIF with proper transparency settings
imageio.mimsave('play_button_click.gif', frames, fps=15, loop=0, palettesize=256, subrectangles=True)

# Save as MP4 (note: MP4 doesn't support transparency)
imageio.mimsave('play_button_click.mp4', frames, fps=15)

print("Animation saved as play_button_click.gif and play_button_click.mp4")
