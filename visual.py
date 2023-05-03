import os
import random
import numpy as np
from math import sqrt, sin, cos, radians
from PIL import Image, ImageDraw, ImageFont, ImageChops
import re
import string
import cairo



def generate_background(img_size):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)

    # Generate random gradient
    gradient = cairo.LinearGradient(0, 0, img_size, img_size)
    gradient.add_color_stop_rgb(0, np.random.rand(), np.random.rand(), np.random.rand())
    gradient.add_color_stop_rgb(1, np.random.rand(), np.random.rand(), np.random.rand())

    # Draw rectangle with gradient
    ctx.rectangle(0, 0, img_size, img_size)
    ctx.set_source(gradient)
    ctx.fill()

    # Create SurfacePattern from surface
    pattern = cairo.SurfacePattern(surface)
    pattern.set_extend(cairo.EXTEND_REPEAT)

    return pattern


def generate_visuals(duration, img_size, num_frames):
    frames = []
    # Generate random background
    background = generate_background(img_size)

    # Create a surface for drawing the shapes
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)


    ctx.paint()

    # Define possible shape types
    shape_types = ['rectangle', 'circle', 'line']

    # Define possible movement types
    movements = ['linear', 'circular', 'bezier']
    # Generate random shapes
    for i in range(num_frames):
        # Draw background
        if random.random() < 0.3:
            ctx.set_source(background)
            ctx.paint()
        # Set random colors
        r, g, b = np.random.rand(3)
        ctx.set_source_rgb(r, g, b)

        # Set random shape parameters
        shape_type = np.random.choice(shape_types)
        if shape_type == 'rectangle':
            x, y = np.random.randint(0, img_size, size=2)
            width, height = np.random.randint(img_size//4, img_size//2, size=2)
            ctx.rectangle(x, y, width, height)
        elif shape_type == 'circle':
            x, y = np.random.randint(0, img_size, size=2)
            radius = np.random.randint(img_size//4, img_size//2)
            ctx.arc(x, y, radius, 0, 2 * np.pi)
        elif shape_type == 'line':
            x1, y1 = np.random.randint(0, img_size, size=2)
            x2, y2 = np.random.randint(0, img_size, size=2)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.set_line_width(np.random.randint(1, 10))
        elif shape_type == 'triangle':
            x1, y1 = np.random.randint(0, img_size, size=2)
            x2, y2 = np.random.randint(0, img_size, size=2)
            x3, y3 = np.random.randint(0, img_size, size=2)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.line_to(x3, y3)
            ctx.line_to(x1, y1)
        elif shape_type == 'ellipse':
            x, y = np.random.randint(0, img_size, size=2)
            width, height = np.random.randint(img_size//4, img_size//2, size=2)
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(width / 2, height / 2)
            ctx.arc(0, 0, 1, 0, 2 * np.pi)
            ctx.restore()
        elif shape_type == 'star':
            x, y = np.random.randint(0, img_size, size=2)
            outer_radius = np.random.randint(img_size//4, img_size//2)
            inner_radius = outer_radius // 2
            num_points = np.random.randint(5, 10)
            angle = np.pi / num_points
            ctx.save()
            ctx.translate(x, y)
            ctx.move_to(outer_radius, 0)
            for i in range(num_points):
                ctx.rotate(angle)
                ctx.line_to(inner_radius, 0)
                ctx.rotate(angle)
                ctx.line_to(outer_radius, 0)
            ctx.restore()

        # Apply random movement to shape
        movement = np.random.choice(movements)
        speed = np.random.randint(1, 10)
        if movement == 'linear':
            ctx.translate(np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2)
        elif movement == 'circular':
            ctx.translate(img_size / 2, img_size / 2)
            ctx.rotate(np.random.random() * 2 * np.pi)
            ctx.translate(-img_size / 2, -img_size / 2)
            ctx.translate(np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2)
        elif movement == 'bezier':
            cp1x, cp1y = np.random.randint(0, img_size, size=2)
            cp2x, cp2y = np.random.randint(0, img_size, size=2)
            x, y = np.random.randint(0, img_size, size=2)
            ctx.curve_to(cp1x, cp1y, cp2x, cp2y, x, y)
        elif movement == 'zigzag':
            x, y = np.random.randint(0, img_size, size=2)
            freq = np.random.randint(5, 20)
            amp = np.random.randint(5, 20)
            offset = np.random.randint(-img_size//4, img_size//4)
            x_offset = (np.sin(i / freq) * amp) + offset
            ctx.translate(x_offset, 0)
        elif movement == 'bounce':
            x, y = np.random.randint(0, img_size, size=2)
            freq = np.random.randint(5, 20)
            amp = np.random.randint(5, 20)
            offset = np.random.randint(-img_size//4, img_size//4)
            y_offset = (np.sin(i / freq) * amp) + offset
            ctx.translate(0, y_offset)

        # Fill or stroke the shape
        if shape_type == 'line':
            ctx.stroke()
        else:
            ctx.fill()

        # Convert surface to PIL image and append to list of frames
        img = Image.frombuffer("RGBA", (img_size, img_size), surface.get_data(), "raw", "BGRA", 0, 1)
        frames.append(img)

    return frames



def generate_random_text():
    inspirational_words = [
        "believe", "dream", "inspire", "achieve", "grow", "create", "motivate", "hope", "strength", "persevere",
        "courage", "love", "kindness", "compassion", "success", "empower", "change", "improve", "dedication",
        "commitment", "confidence", "determination", "positivity", "resilience", "fearless", "unstoppable",
        "progress", "happiness", "gratitude", "mindfulness", "balance", "wisdom", "patience", "adapt", "learn",
        "overcome", "support", "encourage", "collaborate", "transform", "innovate", "endurance", "unity"
    ]

    random_text = " ".join(
        random.choice(inspirational_words)
        for _ in range(random.randint(1, 4))  # Pick 1 to 4 words to reserve slots for "harmony" and "simapsee"
    )

    # Add "harmony" and "simapsee" to the random text
    random_text = f"harmony {random_text} simapsee"

    # Add "get off instagram" or "focus on your goals" to the random text with a certain probability
    if random.random() < 0.3:
        random_text = random.choice(["get off instagram", "focus on your goals"])

    return random_text




def save_instagram_caption(caption, hashtags, filename="instagram_caption.txt"):
    with open(filename, "w") as file:
        file.write(caption)
        file.write("\n\n. . .\n\n")
        file.write(" ".join(hashtags))

def get_random_hashtags(hashtags, num=7):
    return random.sample(hashtags, num)

def generate_title(hashtags, num_words=2):
    words = []
    for hashtag in hashtags:
        words += re.findall(r'\w+', hashtag)

    combined_words = []
    for _ in range(num_words):
        word1 = random.choice(words)
        word2 = random.choice(words)
        word1_part = word1[:len(word1) // 2]
        word2_part = word2[len(word2) // 2:]
        combined_word = word1_part + word2_part
        combined_words.append(combined_word)

    title = " ".join(combined_words) + " " + str(random.randint(1, 999))
    return title

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def generate_fractal_layer(width, height):
    zoom_factor = random.uniform(0.6, 1.5)
    center_x = random.uniform(-2, 0.5)
    center_y = random.uniform(-1.5, 1.5)

    min_x, max_x = center_x - zoom_factor, center_x + zoom_factor
    min_y, max_y = center_y - zoom_factor, center_y + zoom_factor
    max_iter = 1000

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    color_shifts = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))

    for x in range(width):
        for y in range(height):
            real = min_x + (max_x - min_x) * x / (width - 1)
            imag = min_y + (max_y - min_y) * y / (height - 1)
            c = complex(real, imag)
            color = mandelbrot(c, max_iter)
            r = (color * color_shifts[0]) % 256
            g = (color * color_shifts[1]) % 256
            b = (color * color_shifts[2]) % 256
            pixels[x, y] = (r, g, b)

    return img


def draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline=fill)

def draw_tree_of_life(draw, center, radius, fill):
    draw_seed_of_life(draw, center, radius, fill)
    for i in range(3):
        x = center[0] + radius * cos(radians(i * 120))
        y = center[1] + radius * sin(radians(i * 120))
        draw_regular_polygon(draw, (x, y), 6, radius, fill)

def draw_merkaba(draw, center, radius, fill):
    draw_regular_polygon(draw, center, 3, radius, fill)
    draw_regular_polygon(draw, center, 3, radius, fill, rotation=180)

def draw_regular_polygon(draw, center, num_sides, radius, fill, rotation=0):
    angle = 360 / num_sides
    points = []
    for i in range(num_sides):
        x = center[0] + radius * cos(radians(i * angle + rotation))
        y = center[1] + radius * sin(radians(i * angle + rotation))
        points.append((x, y))
    draw.polygon(points, outline=fill)

def draw_vesica_piscis(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    draw_circle(draw, (center[0] + radius, center[1]), radius, fill)

def draw_seed_of_life(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    for i in range(6):
        x = center[0] + radius * cos(radians(i * 60))
        y = center[1] + radius * sin(radians(i * 60))
        draw_circle(draw, (x, y), radius, fill)

def draw_sacred_geometry(img, emblem_size=100):
    center = (img.size[0] // 2, img.size[1] // 2)
    radius = emblem_size // 2
    draw = ImageDraw.Draw(img)
    fill = (255, 255, 255)

    shapes = [
        lambda: draw_circle(draw, center, radius, fill),
        lambda: draw_regular_polygon(draw, center, 3, radius, fill),
        lambda: draw_regular_polygon(draw, center, 4, radius, fill),
        lambda: draw_regular_polygon(draw, center, 5, radius, fill),
        lambda: draw_regular_polygon(draw, center, 6, radius, fill),
        lambda: draw_vesica_piscis(draw, center, radius, fill),
        lambda: draw_seed_of_life(draw, center, radius, fill),
        lambda: draw_tree_of_life(draw, center, radius, fill),
        lambda: draw_merkaba(draw, center, radius, fill),
        lambda: draw_flower_of_life(draw, center, radius, fill),
        lambda: draw_sri_yantra(draw, center, radius, fill),
        lambda: draw_torus(draw, center, radius, fill),
        lambda: draw_metatrons_cube(draw, center, radius, fill),
        lambda: draw_infinity_symbol(draw, center, radius, fill),
        lambda: draw_enneagram(draw, center, radius, fill),
        lambda: draw_tetrahedron(draw, center, radius, fill),
        lambda: draw_icosahedron(draw, center, radius, fill),
        lambda: draw_golden_spiral(draw, center, radius, fill),
        # add more shapes here
    ]

    num_shapes = random.randint(1, 3)
    selected_shapes = random.sample(shapes, num_shapes)
    for shape in selected_shapes:
        shape()
           
                                 
def draw_flower_of_life(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    for i in range(6):
        x = center[0] + radius * cos(radians(i * 60))
        y = center[1] + radius * sin(radians(i * 60))
        draw_circle(draw, (x, y), radius, fill)
        for j in range(5):
            x1 = x + radius * cos(radians(j * 72))
            y1 = y + radius * sin(radians(j * 72))
            draw_circle(draw, (x1, y1), radius, fill)

def draw_sri_yantra(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    for i in range(6):
        x = center[0] + radius * cos(radians(i * 60))
        y = center[1] + radius * sin(radians(i * 60))
        draw_regular_polygon(draw, (x, y), 3, radius / 2, fill, rotation=30)
        for j in range(3):
            x1 = x + radius / 2 * cos(radians(j * 120))
            y1 = y + radius / 2 * sin(radians(j * 120))
            draw_circle(draw, (x1, y1), radius / 6, fill)

    draw_circle(draw, center, radius / 2, fill)
    for i in range(3):
        x = center[0] + radius / 2 * cos(radians(i * 120))
        y = center[1] + radius / 2 * sin(radians(i * 120))
        draw_regular_polygon(draw, (x, y), 3, radius / 6, fill, rotation=30)

    draw_regular_polygon(draw, center, 3, radius / 6, fill)

def draw_torus(draw, center, radius, fill):
    num_points = 50
    points = []
    for i in range(num_points):
        angle = 2 * i * radians(360) / num_points
        x = center[0] + (radius + radius / 2 * cos(angle)) * cos(angle)
        y = center[1] + (radius + radius / 2 * cos(angle)) * sin(angle)
        points.append((x, y))
    draw.polygon(points, outline=fill)

def draw_metatrons_cube(draw, center, radius, fill):
    draw_regular_polygon(draw, center, 3, radius, fill)
    for i in range(3):
        x = center[0] + radius * cos(radians(i * 120))
        y = center[1] + radius * sin(radians(i * 120))
        draw_regular_polygon(draw, (x, y), 4, radius / 2, fill, rotation=45)
        draw_regular_polygon(draw, (x, y), 3, radius / 2 * 3 ** 0.5, fill)

def draw_infinity_symbol(draw, center, radius, fill):
    draw_regular_polygon(draw, center, 3, radius, fill, rotation=30)
    draw_regular_polygon(draw, center, 3, radius, fill, rotation=150)
    draw_circle(draw, center, radius / 3, fill)
    draw_circle(draw, center, radius / 3, fill)

def draw_enneagram(draw, center, radius, fill):
    angle = 360 / 9
    points = []
    for i in range(9):
        x = center[0] + radius * cos(radians(i * angle))
        y = center[1] + radius * sin(radians(i * angle))
        points.append((x, y))
        draw_circle(draw, (x, y), radius / 3, fill)

    for i in range(9):
        draw.line((points[i], points[(i + 3) % 9]), fill=fill)
        draw.line((points[i], points[(i + 6) % 9]), fill=fill)

def draw_tetrahedron(draw, center, radius, fill):
    p = 3 ** 0.5 / 3
    points = [
        (center[0], center[1] - 2 * radius / (3 * p)),
        (center[0] + radius / 2, center[1] + radius / (3 * p)),
        (center[0] - radius / 2, center[1] + radius / (3 * p)),
        (center[0], center[1] + radius / p),
    ]
    draw.polygon(points, outline=fill)

def draw_icosahedron(draw, center, radius, fill):
    p = (1 + 5 ** 0.5) / 2
    points = [
        (center[0], center[1] - radius / p),
        (center[0] - radius / 2, center[1] - 0.5 * radius / p),
        (center[0] + radius / 2, center[1] - 0.5 * radius / p),
        (center[0], center[1] + radius / p),
        (center[0] - radius / p, center[1]),
        (center[0] + radius / p, center[1]),
    ]
    draw.polygon(points, outline=fill)

def draw_golden_spiral(draw, center, radius, fill):
    num_points = 100
    angle = 0
    golden_ratio = (1 + 5 ** 0.5) / 2
    for i in range(num_points):
        angle += radians(360 / num_points)
        r = radius * golden_ratio ** angle
        x = center[0] + r * cos(angle)
        y = center[1] + r * sin(angle)
        draw.point((x, y), fill=fill)


def draw_text(img, text="Focal Point", font_path="Blox2.ttf", font_size=48):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)

    text_x = (img.size[0] - text_width) // 2
    text_y = img.size[1] // 2 + 100

    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
    
def generate_image(width, height, num_layers=3, filename="fractal_image.png"):
    base_image = generate_fractal_layer(width, height)

    for _ in range(num_layers - 1):
        layer = generate_fractal_layer(width, height)
        base_image = ImageChops.blend(base_image, layer, alpha=0.5)

    draw_sacred_geometry(base_image)
    draw_text(base_image)

    base_image.save(filename)