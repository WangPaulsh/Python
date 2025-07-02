import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

# 純Python程式
boysP=[1, 0, 1, 0]
boysQ=[0.4, 0.3, 0.5, 0.8]
girlsP=[0, 1, 0, 0]
girlsQ=[0.3, 0.4, 0.2, 0.1]
othersP=[0, 0, 0, 1]
othersQ=[0.3, 0.3, 0.3, 0.1]

boysCross=0
for i in range(len(boysP)):
    boysCross+=-(boysP[i]*np.log2(boysQ[i]))
print("Cross-entropy boy of P on Q is: ", boysCross)
girlsCross=0
for i in range(len(girlsP)):
    girlsCross+=-(girlsP[i]*np.log2(girlsQ[i]))
print("Cross-entropy girl of P on Q is: ", girlsCross)
OthersCross=0
for i in range(len(othersP)):
    OthersCross+=-(othersP[i]*np.log2(othersQ[i]))
print("Cross-entropy other of P on Q is: ", OthersCross)

totalCross = boysCross + girlsCross + OthersCross

# 只計算total cross-entropy
P=[boysP, girlsP, othersP]
Q=[boysQ, girlsQ, othersQ]
totalCross=0
for i in range(len(P)):
    for j in range(len(P[i])):
        totalCross+=-(P[i][j]*np.log2(Q[i][j]))

print("Total cross-entropy is: ", totalCross)


# 使用NumPy計算
P_labels = np.array([0, 1, 0, 2])
num_classes = 3
ohl_P = np.eye(num_classes)[P_labels]
print(ohl_P)
Q_predictions = np.array([
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.5, 0.2, 0.3],
    [0.8, 0.1, 0.1]
])
# 為了避免 log(0) 出現，通常會加一個小常數
epsilon = 1e-12
Q_predictions = np.clip(Q_predictions, epsilon, 1. - epsilon)
total_cross_entropy = -np.sum(ohl_P * np.log2(Q_predictions))
print("Cross Entropy:", total_cross_entropy)


# 使用PyTorch計算
P_labels = torch.tensor([0, 1, 0, 2])
Q_predictions = torch.tensor([
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.5, 0.2, 0.3],
    [0.8, 0.1, 0.1]
])

# 直接計算
P_ohl_labels = torch.nn.functional.one_hot(P_labels, num_classes=3).float()
loss_manual = -torch.sum(P_ohl_labels * torch.log2(Q_predictions))
print("Cross Entropy Loss:", loss_manual.item())

# 手動取正確類別的預測機率
picked_probs = Q_predictions[torch.arange(len(P_labels)), P_labels]
# 從機率推回logits
picked_probs_logits = torch.log2(picked_probs)
cross_entropy = -torch.sum(picked_probs_logits)
print("Total Cross Entropy (Manual PyTorch):", cross_entropy.item())


# 使用TensorFlow計算
P_labels = tf.constant([0, 1, 0, 2])
Q_predictions = tf.constant([
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.5, 0.2, 0.3],
    [0.8, 0.1, 0.1]
])

# # 直接計算
P_ohl_labels = tf.one_hot(P_labels, depth=3)
loss_manual = -tf.reduce_sum(P_ohl_labels * (tf.math.log(Q_predictions))/tf.math.log(2.0))
print("Manual Cross Entropy Loss:", loss_manual.numpy())

# 手動計算 cross entropy
# 取出每一筆樣本中「正確類別」對應的預測機率
picked_probs = tf.gather_nd(Q_predictions, indices=tf.stack([tf.range(len(P_labels)), P_labels], axis=1))
# 取 log (換底公式轉換為log2)
picked_probs_logits = tf.math.log(picked_probs)/tf.math.log(2.0)
# cross entropy 是 -log(正確類別的預測機率)，然後加總
cross_entropy = -tf.reduce_sum(picked_probs_logits)
print("Total Cross Entropy (Manual TensorFlow):", cross_entropy.numpy())