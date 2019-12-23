import requests
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from io import BytesIO

import colorscheme


TEST = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1i6jN4Lj2kP2glv4vD3p16chAH6q-V9JpeHYd6URd9-GyoM7reg'

test = colorscheme.colorscheme(requests.get(TEST).content)
plt.imshow(np.array(Image.open(BytesIO(test))))
plt.show()