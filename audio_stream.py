import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg

import numpy as np
import pyaudio
import struct
import matplotlib.pyplot as plt
from logger_info import setup_logging
from constants import *

# Setup logging
logger = setup_logging()

# Create a PyAudio object
p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# read one chunk
data = stream.read(CHUNK) # data is a byte string | b'\xa4\xfb\xdd\
# print(data)
logger.info(len(data)) # 4096 * 2 = 8192 bytes
logger.info(len(data)*2) 

# integer data
data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
logger.info(data_int) # data from 0 to 255

# plot the data
fig, ax = plt.subplots()
ax.plot(data_int, '-')
plt.show() 
# it shows a kind of clipped or split signal