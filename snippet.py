import matplotlib.pyplot as plt
import numpy as np
import torch
a = torch.zeros((2000,1500))
plt.figure(1)
plt.imshow(a.numpy())
plt.figure(2)
a[:,50:200] = 1
plt.imshow(a.numpy())
plt.show()