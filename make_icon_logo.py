import sys
from PIL import Image, ImageDraw, ImageFont

def make_icon_logo(size=512, output_path=None):
    if output_path is None:
        output_path = rf"C:\Users\Adam Cheng\Documents\Pictures\DocTranslator_Logo_{size}.jpg"
    
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    blue = (37, 99, 235)
    
    # Circle
    circle_radius = int(size * 0.32)
    cx, cy = size // 2, size // 2
    draw.ellipse(
        [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
        fill=blue
    )
    
    # Document icon inside the circle
    # Paper shape with folded corner
    doc_w = int(circle_radius * 1.0)
    doc_h = int(doc_w * 1.3)
    doc_left = cx - doc_w // 2
    doc_top = cy - doc_h // 2
    doc_right = doc_left + doc_w
    doc_bottom = doc_top + doc_h
    
    # Main paper body (white)
    corner_fold = int(doc_w * 0.3)
    
    # Draw main rectangle
    draw.rounded_rectangle(
        [doc_left, doc_top, doc_right - corner_fold, doc_bottom],
        radius=int(size * 0.02),
        fill=(255, 255, 255)
    )
    
    # Draw the folded corner triangle
    fold_points = [
        (doc_right - corner_fold, doc_top),
        (doc_right, doc_top + corner_fold),
        (doc_right - corner_fold, doc_top + corner_fold)
    ]
    draw.polygon(fold_points, fill=(220, 225, 235))  # slightly darker for fold
    
    # Draw the remaining part of the paper (right side below fold)
    draw.rectangle(
        [doc_right - corner_fold, doc_top + corner_fold, doc_right, doc_bottom],
        fill=(255, 255, 255)
    )
    
    # Draw fold line
    draw.line(
        [(doc_right - corner_fold, doc_top), (doc_right - corner_fold, doc_top + corner_fold), (doc_right, doc_top + corner_fold)],
        fill=(180, 190, 210),
        width=max(1, int(size * 0.008))
    )
    
    # Text lines inside document (simulating text)
    line_margin = int(doc_w * 0.15)
    line_y_start = doc_top + int(doc_h * 0.22)
    line_spacing = int(doc_h * 0.13)
    line_width = doc_w - line_margin * 2
    
    for i in range(4):
        line_y = line_y_start + i * line_spacing
        # Vary line lengths for realism
        if i == 3:
            lw = int(line_width * 0.6)
        else:
            lw = line_width
        draw.rectangle(
            [doc_left + line_margin, line_y, doc_left + line_margin + lw, line_y + int(doc_h * 0.04)],
            fill=(220, 225, 235)
        )
    
    # "DocTranslator" below the circle
    font_size = int(size * 0.10)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = "DocTranslator"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw//2, cy + circle_radius + int(size * 0.04)), text, fill=blue, font=font)
    
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
    make_icon_logo(512)
    make_icon_logo(256)
    make_icon_logo(64)
