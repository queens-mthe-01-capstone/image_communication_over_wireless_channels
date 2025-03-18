
def crop_to_square(image):
  height, width = image.shape[:2]
  min_side = min(height, width)
  start_x = (width - min_side) // 2
  start_y = (height - min_side) // 2
  cropped_image = image[start_y:start_y + min_side, start_x:start_x + min_side]
  return cropped_image