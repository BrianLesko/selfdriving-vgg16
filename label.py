# Brian Lesko
# 1/19/25 
# use Moondream to label images for robot steering program. 

import os
from PIL import Image
import torch
import streamlit as st
import cv2
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
import moondream as md
import time
from PIL import Image, ImageDraw

# UI Setup
st.title("Labelling")
ImageSpot = st.empty()
TimeSpot = st.empty()
Result = st.empty()

# Moondream initialization
model_id = "vikhyatk/moondream2"
revision = "2025-01-09"  # Specify revision for stability
device = "mps"
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  revision=revision,
  trust_remote_code=True,
  torch_dtype= torch.float64,
  #low_cpu_mem_usage=True,
  device_map={"": device}
)
model = model.to(device)

# Take a folder of images
valid_extensions = {".jpg", ".jpeg", ".png"}
source_dir = "labelled_dataset_backup"

# Check if a given file is an image
def is_image_file(filename):
  return any(filename.lower().endswith(ext) for ext in valid_extensions)

def annotate_points(image, points, color=(255, 255, 255, 160)):
  # Define overlay and drawing parameters
  width, height = image.size
  radius1 = 10
  radius = 11
  radius2 = 3
  overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
  draw = ImageDraw.Draw(overlay)

  # Draw points on overlay
  #for point in points:
  if True:
    point = points[0]
    point = (int(point['x']*width), int(point['y']*height))
    draw.ellipse([point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius], fill=color)
    draw.ellipse([point[0]-radius1, point[1]-radius1, point[0]+radius1, point[1]+radius1], fill=(255, 255, 255, 0)) # Clear center
    draw.ellipse([point[0]-radius2, point[1]-radius2, point[0]+radius2, point[1]+radius2], fill=color)
  result = Image.alpha_composite(image, overlay)
  return result

def annotate_box(image, objects, color=(255, 255, 255, 160)):
  width, height = image.size
  overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
  draw = ImageDraw.Draw(overlay)

  #for object in objects:
  #  draw.rectangle(xy=object['x_min'])

left_folder = os.path.join(source_dir, "right")
left = 0
right = 0
for img_name in filter(is_image_file, os.listdir(left_folder)):
  img_path = os.path.join(left_folder, img_name)
  with Image.open(img_path) as img:
    time1 = time.time()
    # now we have the image open, we can do whatever we want with it

    # Display the image
    #ImageSpot.image(img, caption=f"{img_name}")
    # Moondream labelling
    encoded_image = model.encode_image(img)
    img = img.convert("RGBA")
    #model.detect(img, "Obstacle")["objects"]

    # Points to avoid
    x1, x2, x3 = .5, .5, .5
    points2 = model.point(encoded_image, "Cliff")["points"]
    if points2:
      x2 = points2[0]['x']
      img = annotate_points(img, points2, color=(255, 0, 0, 160))
    points3 = model.point(encoded_image, "Wall")["points"]
    if points3:
      x3 = points3[0]['x']
      img = annotate_points(img, points3, color=(255, 0, 0, 160))
    points1 = model.point(encoded_image, "Barrier")["points"]
    if points1:
      x1 = points1[0]['x']
      img = annotate_points(img, points1, color=(255, 0, 0, 160))
    x_avg1 = (x1 + x2 + x3) /3
    x_avg1 = -x_avg1 + 1

    # Points to go towards
    x1, x2, x3 = .5, .5, .5
    points1 = model.point(encoded_image, "Center of the path")["points"]
    if points1:
      x1 = points1[0]['x']
      img = annotate_points(img, points1, color=(0, 255, 0, 160))
    points2 = model.point(encoded_image, "Center of the hallway")["points"]
    if points2:
      x2 = points2[0]['x']
      img = annotate_points(img, points2, color=(0, 255, 0, 160))
    x_avg2 = (x1 + x2) /2
    
    x_avg = (x_avg1+x_avg2) -1 
    if x_avg < 0:
      left=left+1
    else:
      right=right+1

    Result.write(f"Left: {left} Right: {right}")

    ImageSpot.image(img, caption="Labelled Image")
    TimeSpot.text(f"Time taken: {time.time() - time1}")
