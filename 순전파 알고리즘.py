

# =============================================================================
# Author : Gyeom
# =============================================================================

# =============================================================================
# Step1) Propagation
# =============================================================================
import numpy as np


# 잠시 혼자 놀기
A = np.array([['가','나','다','라'],
              ['a','b','c','d'],
              [1,2,3,4]])

A.T


# 사용자 정의 ReLU 함수
def ReLU(x):
    return np.maximum(x, 0)


x = [1.3, 0.8]
w1 = [[0.1, 0.2, 0.3],
      [-0.1, 0.0, 0.1]]
# w1 = np.random.random(size=(2,3))


X = np.array(x)
X_T = X.T
W1 = np.array(w1)

fst_layer = np.matmul(X,W1)    
fst_layer_activ = ReLU(fst_layer)


W2 = np.array([[0.1, 0.2, 0.3],
                [0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1]])
# W2 = np.random.random(size=(3,3))

sec_layer = np.matmul(fst_layer_activ, W2)
sec_activ = ReLU(sec_layer)


W3 = np.array([0.2, 0.1, 0.0])
# W3 = np.random.random(size=(1,3))

predicition = np.matmul(sec_activ, W3)

ground_truth = 0.5

error = (ground_truth-predicition) ** 2
print(round(error,4))



# =============================================================================
# Step2) Backpropagation
# =============================================================================




