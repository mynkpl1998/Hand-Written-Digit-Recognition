import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('Data/test.csv')
data = np.array(df)

for x in range(0,len(data)):
	img_matrix = np.split(data[x],28)
	
	plt.imshow(img_matrix,interpolation='nearest')
	plt.axis('off')
	
	plt.savefig('Test_Input/image_'+str(x)+'.png')
