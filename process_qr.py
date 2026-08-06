import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def process_qr():
    qr_path = r"C:\Users\Adam Cheng\Documents\Pictures\QR Code for Receiving Money.jpg"
    output_path = r"C:\Users\Adam Cheng\Documents\Pictures\QR Code with DocTranslator Logo.jpg"
    
    # Load QR code
    qr = Image.open(qr_path).convert('RGBA')
    w, h = qr.size
    
    # WeChat QR code center photo region is typically ~30-35% of image size
    # The photo is in a rounded rectangle with white border
    region_size = int(min(w, h) * 0.34)
    left = (w - region_size) // 2
    top = (h - region_size) // 2
    right = left + region_size
    bottom = top + region_size
    
    # Create a rounded rectangle mask for the photo region
    corner_radius = int(region_size * 0.12)
    mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([left, top, right, bottom], radius=corner_radius, fill=255)
    
    # Save the WeChat green icon area (bottom-right corner of photo region)
    icon_size = int(region_size * 0.25)
    icon_left = right - icon_size - int(region_size * 0.04)
    icon_top = bottom - icon_size - int(region_size * 0.04)
    icon_region = qr.crop((icon_left, icon_top, icon_left + icon_size, icon_top + icon_size))
    
    # Create the DocTranslator logo to overlay in the center region
    logo = Image.new('RGB', (region_size, region_size), (255, 255, 255))
    logo_draw = ImageDraw.Draw(logo)
    
    # Create a rounded rectangle mask for the logo shape
    logo_mask = Image.new('L', (region_size, region_size), 0)
    logo_mask_draw = ImageDraw.Draw(logo_mask)
    logo_mask_draw.rounded_rectangle([0, 0, region_size, region_size], radius=corner_radius, fill=255)
    
    # Blue circle in the center (DocTranslator brand color)
    circle_radius = int(region_size * 0.32)
    cx, cy = region_size // 2, region_size // 2
    logo_draw.ellipse(
        [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
        fill=(37, 99, 235)
    )
    
    # Try to load a nice font, fallback to default
    try:
        font_doc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(region_size * 0.15))
        font_translator = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(region_size * 0.09))
        font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(region_size * 0.06))
    except:
        font_doc = ImageFont.load_default()
        font_translator = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw "Doc" in white on the blue circle
    text = "Doc"
    bbox = logo_draw.textbbox((0, 0), text, font=font_doc)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    logo_draw.text((cx - tw//2, cy - th//2 - 2), text, fill=(255, 255, 255), font=font_doc)
    
    # Draw "Translator" below the circle
    text2 = "Translator"
    bbox2 = logo_draw.textbbox((0, 0), text2, font=font_translator)
    tw2 = bbox2[2] - bbox2[0]
    th2 = bbox2[3] - bbox2[1]
    logo_draw.text((cx - tw2//2, cy + circle_radius + int(region_size * 0.04)), text2, fill=(37, 99, 235), font=font_translator)
    
    # Add a subtle border
    border_width = int(region_size * 0.02)
    logo_draw.rounded_rectangle([border_width, border_width, region_size - border_width, region_size - border_width], 
                                 radius=corner_radius - border_width, 
                                 outline=(37, 99, 235), width=border_width)
    
    # Create result image
    result = qr.copy()
    
    # Paste logo into the center region using the rounded rectangle mask
    # The mask for paste must match the size of the source image (logo)
    result.paste(logo, (left, top), logo_mask)
    
    # Paste the WeChat green icon back on top
    result.paste(icon_region, (icon_left, icon_top))
    
    # Save
    result_rgb = result.convert('RGB')
    result_rgb.save(output_path, quality=95)
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    process_qr()
