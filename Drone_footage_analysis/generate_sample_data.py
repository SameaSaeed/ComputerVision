import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
import random

def generate_healthy_crop_image(size=(224, 224)):
    """Generate synthetic healthy crop image"""
    # Create base green image
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:, :, 1] = np.random.randint(100, 180, size)  # Green channel
    img[:, :, 0] = np.random.randint(20, 80, size)    # Blue channel
    img[:, :, 2] = np.random.randint(20, 80, size)    # Red channel
    
    # Add some texture
    noise = np.random.randint(-20, 20, size)
    img = np.clip(img + noise, 0, 255)
    
    return img

def generate_diseased_crop_image(size=(224, 224)):
    """Generate synthetic diseased crop image"""
    # Start with healthy crop
    img = generate_healthy_crop_image(size)
    
    # Add brown/yellow spots for disease
    for _ in range(random.randint(3, 8)):
        center = (random.randint(20, size[0]-20), random.randint(20, size[1]-20))
        radius = random.randint(10, 30)
        cv2.circle(img, center, radius, (30, 100, 150), -1)  # Brown spots
    
    return img

def generate_bare_soil_image(size=(224, 224)):
    """Generate synthetic bare soil image"""
    # Brown/tan colors for soil
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:, :, 0] = np.random.randint(80, 120, size)   # Blue
    img[:, :, 1] = np.random.randint(100, 140, size)  # Green
    img[:, :, 2] = np.random.randint(120, 160, size)  # Red
    
    # Add texture
    noise = np.random.randint(-30, 30, size)
    img = np.clip(img + noise, 0, 255)
    
    return img

def generate_weed_image(size=(224, 224)):
    """Generate synthetic weed image"""
    # Darker, more chaotic green
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:, :, 1] = np.random.randint(60, 120, size)   # Green
    img[:, :, 0] = np.random.randint(10, 50, size)    # Blue
    img[:, :, 2] = np.random.randint(10, 50, size)    # Red
    
    # Add random patches
    for _ in range(random.randint(5, 15)):
        x, y = random.randint(0, size[0]-10), random.randint(0, size[1]-10)
        w, h = random.randint(5, 20), random.randint(5, 20)
        color = (random.randint(0, 100), random.randint(80, 150), random.randint(0, 100))
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    
    return img

def create_dataset():
    """Create complete sample dataset"""
    categories = {
        'healthy_crops': generate_healthy_crop_image,
        'diseased_crops': generate_diseased_crop_image,
        'bare_soil': generate_bare_soil_image,
        'weeds': generate_weed_image
    }
    
    samples_per_category = 50
    
    for category, generator_func in categories.items():
        print(f"Generating {samples_per_category} images for {category}...")
        
        for i in range(samples_per_category):
            img = generator_func()
            filename = f"data/{category}/sample_{i:03d}.jpg"
            cv2.imwrite(filename, img)
        
        print(f"✓ {category}: {samples_per_category} images created")
    
    print("\nDataset creation completed!")
    print(f"Total images: {len(categories) * samples_per_category}")

if __name__ == "__main__":
    create_dataset()