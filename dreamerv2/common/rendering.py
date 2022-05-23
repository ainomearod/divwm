import numpy as np

TILE_PIXELS = 32


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0]//factor, factor, img.shape[1]//factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img

def render_tile(
  obj,
  tile_size=TILE_PIXELS,
  subdivs=3
):
  """
  Render a tile and cache the result
  """

  img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

  if obj != None and obj.type != 'floor':
      obj.render(img)

  # Downsample the image to perform supersampling/anti-aliasing
  img = downsample(img, subdivs)

  return img

def render_minigrid(env):
  ## Adapted from https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py

  ## Render the whole grid
  ## hardcode tile_size to match the default https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py#L11
#   env.grid.render_tile = render_tile
  img = env.grid.render(tile_size=TILE_PIXELS)
  goal = (0, 0)

  ## Hack: convert grey to black
  for j in range(env.grid.height):
      for i in range(env.grid.width):
          obj = env.grid.get(i, j)
          if obj is not None:
            # print(obj.type)
            if obj.type == 'floor':
                # print("floor")
                ymin = j * TILE_PIXELS
                ymax = (j+1) * TILE_PIXELS
                xmin = i * TILE_PIXELS
                xmax = (i+1) * TILE_PIXELS
                img[ymin:ymax, xmin:xmax, :].fill(255)
            elif obj.type == 'wall':
                # print("wall")
                ymin = j * TILE_PIXELS
                ymax = (j+1) * TILE_PIXELS
                xmin = i * TILE_PIXELS
                xmax = (i+1) * TILE_PIXELS
                img[ymin:ymax, xmin:xmax, :].fill(0)
            elif obj.type == 'goal':
                ymin = j * TILE_PIXELS
                ymax = (j+1) * TILE_PIXELS
                xmin = i * TILE_PIXELS
                xmax = (i+1) * TILE_PIXELS
                img[ymin:ymax, xmin:xmax, :] = [255, 0, 0]
                goal = (j, i)
            elif obj.type not in ['key', 'ball', 'box', 'goal', 'lava', 'door']:
                ymin = j * TILE_PIXELS
                ymax = (j+1) * TILE_PIXELS
                xmin = i * TILE_PIXELS
                xmax = (i+1) * TILE_PIXELS
                img[ymin:ymax, xmin:xmax, :].fill(255)
          else:
            ymin = j * TILE_PIXELS
            ymax = (j+1) * TILE_PIXELS
            xmin = i * TILE_PIXELS
            xmax = (i+1) * TILE_PIXELS
            img[ymin:ymax, xmin:xmax, :].fill(255)

#   for y in range(img.shape[0]):
#       for x in range(img.shape[1]):
#           if (img[y, x] == np.array([100, 100, 100])).all():

#               img[y, x] = np.array([255, 255, 255])
  img[img == 100] = 255
  orig_img = env.render(highlight=False)

  return img, goal, orig_img