import matplotlib.pyplot as plt
import numpy as np

from curvefitting_assessor import CurvefittingAssessor
from curvefunctions import *
from model_factory import *

# best_performance = 0.9734


x1 = range(1, 8)
y1 = [0.1010, 0.1028, 0.1028, 0.1135, 0.0958, 0.1010, 0.1010]
# predict_y = 0.7819696259575892
# Time: 50s
# learning curve is incresing, least_sq lower slowly
'''
x1 = range(1, 7)
y1 = [0.1010, 0.1010, 0.1009, 0.1010, 0.1134, 0.0958]
xx = range(1, 22)
yy = [0.1010, 0.1010, 0.1009, 0.1010, 0.1134, 0.0958, 0.0958, 0.0974, 0.1135, 0.1135, 0.098, 0.1009, 0.1032, 0.1135, 0.0958, 0.1032, 0.0958, 0.1135, 0.0958, 0.1010, 0.1028]
'''

plt.figure()
plt.scatter(x1[:], y1[:], 25, "green")
#plt.scatter(xx[:], yy[:], 25, "blue")

assessor = CurvefittingAssessor(threshold=0.9)
assessor.completed_best_performance = 0.956300
assessor.set_best_performance = True

print ('id = 1', assessor.assess_trial(1, y1))
assessor.trial_end(1, True)

#print ('id = 2', assessor.assess_trial(2, y2))
#assessor.trial_end(2, True)

plotx = np.arange(0, 25, 0.01)
for model in ddd:
    if model_para_num[model] == 2:
        ploty = all_models[model](plotx, model_para[model][0], model_para[model][1])
    elif model_para_num[model] == 3:
        ploty = all_models[model](plotx, model_para[model][0], model_para[model][1], model_para[model][2])
    elif model_para_num[model] == 4:
        ploty = all_models[model](plotx, model_para[model][0], model_para[model][1], model_para[model][2], model_para[model][3])
    plt.plot(plotx, ploty, 'red')

plt.show()
