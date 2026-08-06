import sys
from PIL import Image, ImageDraw, ImageFont

def make_logo(size=512, output_path=None):
    if output_path is None:
        output_path = rf"C:\Users\Adam Cheng\Documents\Pictures\DocTranslator_Logo_{size}.jpg"
    
    # Create white background
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # DocTranslator Classic Blue
    blue = (37, 99, 235)
    
    # Circle size
    circle_radius = int(size * 0.32)
    cx, cy = size // 2, size // 2
    
    # Blue circle
    draw.ellipse(
        [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
        fill=blue
    )
    
    # Fonts
    font_size_doc = int(size * 0.18)
    font_size_translator = int(size * 0.10)
    font_size_line = int(size * 0.05)
    
    try:
        font_doc = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size_doc)
        font_translator = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size_translator)
        font_line = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size_line)
    except:
        try:
            font_doc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size_doc)
            font_translator = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size_translator)
            font_line = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size_line)
        except:
            font_doc = ImageFont.load_default()
            font_translator = ImageFont.load_default()
            font_line = ImageFont.load_default()
    
    # "Doc" in white centered in the blue circle
    text = "Doc"
    bbox = draw.textbbox((0, 0), text, font=font_doc)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw//2, cy - th//2 - 4), text, fill=(255, 255, 255), font=font_doc)
    
    # "Translator" below the circle
    text2 = "Translator"
    bbox2 = draw.textbbox((0, 0), text2, font=font_translator)
    tw2 = bbox2[2] - bbox2[0]
    th2 = bbox2[3] - bbox2[1]
    draw.text((cx - tw2//2, cy + circle_radius + int(size * 0.04)), text2, fill=blue, font=font_translator)
    
    # Rounded border
    border_width = max(2, int(size * 0.02))
    corner_r = int(size * 0.12)
    draw.rounded_rectangle(
        [border_width, border_width, size - border_width, size - border_width],
        radius=corner_r,
        outline=blue,
        width=border_width
    )
    
    img.save(output_path, quality=95)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    # Generate a few useful sizes
    make_logo(512)  # Large - for print/pricing page
    make_logo(256)  # Medium - for web
    make_logo(64)   # Small - for favicon/icon
