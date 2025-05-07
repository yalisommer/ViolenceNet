import numpy as np

true_violent1 = np.array([
    1.0, 1.0, 1.0, 0.7846, 1.0,
    1.0, 0.9976, 1.0, 0.99997, 1.0,
    0.8516, 0.9999, 0.9764, 1.0, 1.0,
    1.0, 0.99999988, 0.99999833, 1.0, 1.0
])

true_nonviolent1 = np.array([
    0.9714, 1.0, 0.9949, 0.8638, 0.9991,
    0.9972, 0.9983, 0.5883, 0.6989, 0.6399,
    0.9995, 0.9847, 1.0, 0.9993, 0.5433,
    1.0, 0.99999, 0.8570, 0.9988, 1.0
])

true_violent = np.array([
    0.9904, 0.9866, 0.9953, 0.9648, 0.8527,
    0.9271, 0.7684, 0.6727, 0.9185, 0.9074,
    0.8076, 0.6249, 0.5317, 0.8693, 0.9779,
    0.8221, 0.6845, 0.6417, 0.9165, 0.9764
])

true_nonviolent = np.array([
    0.9648, 0.9589, 0.9810, 0.9418, 0.9883,
    0.9039, 0.9295, 0.7687, 0.6168, 0.8714,
    0.9744, 0.9807, 0.9555, 0.7525, 0.8818,
    0.8047, 0.9542, 0.6194, 0.7581, 0.7239
])

thresholds = np.linspace(0.5, 1.0, 100)
correct_counts = []

for thresh in thresholds:
    tp = np.sum(true_violent1 >= thresh)      # correctly predicted violent
    tn = np.sum(true_nonviolent1 < thresh)    # correctly predicted non-violent
    correct = tp + tn
    correct_counts.append(correct)

max_correct = max(correct_counts)
best_thresholds = thresholds[np.where(correct_counts == max_correct)]

print("best threshold: " + str(best_thresholds[0]))
print("correct predictions: " + str(max_correct))