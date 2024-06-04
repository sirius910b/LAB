

# =============================================================================
# Author : Gyeom
# =============================================================================


import numpy as np

# 사용자 정의 ReLU 함수
def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


# input 데이터 : 3x4
X = np.array([[80,89,70,86],
              [96,98,90,82],
              [100,80,82,76]])

Y = np.array([[81.25], [91.5], [84.5]])


def CNA(X,W,active_func):
    # CNA = Calculate Node and Activation
    Node =  np.matmul(X,W)
    Activated = active_func(Node)
    return Activated


# =============================================================================
# 1행 데이터로 테스트 한번 하기
# =============================================================================
# # 1행 데이터
# X1 = X[0].T


# # W1 : 노드는 6개정도?
# W1 = np.random.random(size=(4,6))

# Layer_1st_activated = CNA(X1, W1, ReLU)

# # W2 : 노드는 5개
# W2 = np.random.random(size=(6,5))

# Layer_2nd_activated = CNA(Layer_1st_activated, W2, ReLU)

# # 3층
# W3 = np.random.random(size=(5,3))

# Layer_3rd_activated = CNA(Layer_2nd_activated, W3, ReLU)


# # 마지막층
# W4 = np.random.random(size=(3,1))

# y_predict = np.matmul(Layer_3rd_activated, W4)


# # 오류(Error) 측정
# Error_1st = y_predict - Y[0]







# =============================================================================
# 본격 순전파 구현
# =============================================================================
# 신경망 가중치 초기화 (한 번만 초기화)
np.random.seed(0)  # 가중치 초기화를 위한 시드 설정
W1 = np.random.random(size=(4, 6))
W2 = np.random.random(size=(6, 5))
W3 = np.random.random(size=(5, 3))
W4 = np.random.random(size=(3, 1))

Error_list = []


for i, x in enumerate(X):
    # 첫 번째 층
    Layer_1st_activated = CNA(x, W1, ReLU)

    # 두 번째 층
    Layer_2nd_activated = CNA(Layer_1st_activated, W2, ReLU)

    # 세 번째 층
    Layer_3rd_activated = CNA(Layer_2nd_activated, W3, ReLU)

    # 마지막 층 (회귀 문제이므로 활성화 함수 없음)
    y_predict = np.matmul(Layer_3rd_activated, W4)

    # 오차(Error) 측정
    Error = y_predict - Y[i]
    
    Error_list.append(Error)


Error_mean = np.array(Error_list).mean()
RMS_error = np.sqrt(np.mean(np.array(Error_list)**2))
print("RMS Error:", RMS_error)




# =============================================================================
# 순전파 + 역전파 구현
# =============================================================================
import numpy as np

# 사용자 정의 ReLU 함수
def ReLU(x):
    return np.maximum(x, 0)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# input 데이터 : 3x4
X = np.array([[80,89,70,86],
              [96,98,90,82],
              [100,80,82,76]])

Y = np.array([[81.25], [91.5], [84.5]])

def CNA(X, W, active_func):
    # CNA = Calculate Node and Activation
    Node = np.dot(X, W)
    Activated = active_func(Node)
    return Node, Activated

# 신경망 가중치 초기화 (한 번만 초기화)
np.random.seed(0)  # 가중치 초기화를 위한 시드 설정
W1 = np.random.random(size=(4, 6))
W2 = np.random.random(size=(6, 5))
W3 = np.random.random(size=(5, 3))
W4 = np.random.random(size=(3, 1))

learning_rate = 0.001
epochs = 10000

for epoch in range(epochs):
    Error_list = []

    for i, x in enumerate(X):
        # 순전파 (Forward Pass)
        Node_1, Layer_1st_activated = CNA(x, W1, ReLU)
        Node_2, Layer_2nd_activated = CNA(Layer_1st_activated, W2, ReLU)
        Node_3, Layer_3rd_activated = CNA(Layer_2nd_activated, W3, ReLU)
        Node_4 = np.dot(Layer_3rd_activated, W4)
        y_predict = Node_4
        
        # 오차 계산
        Error = y_predict - Y[i]
        Error_list.append(Error)
        
        # 역전파 (Backpropagation)
        dL_dy = 2 * Error  # MSE의 미분
        dL_dW4 = np.dot(Layer_3rd_activated[:, np.newaxis], dL_dy[np.newaxis, :])
        
        dL_dLayer_3rd = np.dot(dL_dy, W4.T) * ReLU_derivative(Node_3)
        dL_dW3 = np.dot(Layer_2nd_activated[:, np.newaxis], dL_dLayer_3rd[np.newaxis, :])
        
        dL_dLayer_2nd = np.dot(dL_dLayer_3rd, W3.T) * ReLU_derivative(Node_2)
        dL_dW2 = np.dot(Layer_1st_activated[:, np.newaxis], dL_dLayer_2nd[np.newaxis, :])
        
        dL_dLayer_1st = np.dot(dL_dLayer_2nd, W2.T) * ReLU_derivative(Node_1)
        dL_dW1 = np.dot(x[:, np.newaxis], dL_dLayer_1st[np.newaxis, :])
        
        # 가중치 업데이트
        W4 -= learning_rate * dL_dW4
        W3 -= learning_rate * dL_dW3
        W2 -= learning_rate * dL_dW2
        W1 -= learning_rate * dL_dW1

    if epoch % 1000 == 0:
        Error_mean = np.array(Error_list).mean()
        RMS_error = np.sqrt(np.mean(np.array(Error_list)**2))
        print(f'Epoch {epoch}, Error_mean: {Error_mean}')
        print(f"Epoch {epoch}, RMS Error: {RMS_error}")

# 최종 RMS Error 출력
Error_mean = np.array(Error_list).mean()
RMS_error = np.sqrt(np.mean(np.array(Error_list)**2))
print("Final RMS Error:", RMS_error)






