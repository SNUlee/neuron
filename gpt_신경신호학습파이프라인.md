아래는 신경신호가 저장된 HDF5(HDF) 파일들을 활용하여 학습용 데이터셋을 구성하고, PyTorch를 이용해 모델을 학습 및 평가하는 전 과정을 MLflow로 관리하는 전체적인 예시 워크플로우입니다. 여기서 제시하는 코드는 개념적 예시이며, 실제 신호 처리나 모델 구조는 상황에 맞게 수정해야 합니다. 또한 GPU 환경 사용, 최신 패키지 사용, mlflow 추적, streamlit을 통한 간단한 데이터 시각화 예제를 함께 포함합니다. 실제 환경에 맞추어 파일 경로, 데이터 구조 등을 조정하시길 바랍니다.

## 전체 처리 프로세스 개요

1. **데이터 로드 및 전처리**:  
   - HDF5 파일(h5py)에서 신경신호 데이터를 로드하고, 라벨(또는 타겟)과 함께 PyTorch `Dataset` 형태로 구성합니다.  
   - 데이터 전처리(정규화, 필요하다면 필터링 등) 진행.
   
2. **데이터셋 분할**:  
   - 훈련용(train), 검증용(validation), 테스트용(test) 데이터셋을 분리합니다.
   
3. **모델 정의**:  
   - PyTorch로 신경신호 특성에 맞는 모델(예: 1D CNN, RNN, Transformer 등)을 정의합니다. 여기서는 간단한 MLP 예시를 듭니다.
   
4. **학습 및 평가 파이프라인 구성**:  
   - `DataLoader`를 이용해 배치별 데이터 로딩.
   - GPU 사용 환경 고려(`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
   - 손실함수, 옵티마이저 설정 후 학습 루프 작성.
   
5. **MLflow를 통한 실험 관리**:  
   - mlflow run 시작 (`mlflow.start_run()`)
   - 학습 과정에서 loss, accuracy 등 메트릭 로깅
   - 모델 가중치(artifact)와 파라미터 로깅
   - 학습 완료 후 테스트 평가 및 결과 로깅
   
6. **Streamlit을 통한 간단한 데이터 시각화**:  
   - Streamlit 앱을 통해 HDF5로부터 로드한 신경신호 일부 샘플을 plot하여 데이터 특성 파악.

## 예시 코드 스니펫

아래 코드는 하나의 `train.py` 스크립트로 가정한 예시입니다. 실제 실행 시 프로젝트 구조와 파일 경로 등을 조정하십시오.

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch
import streamlit as st
import matplotlib.pyplot as plt

#######################
# 1. 데이터셋 정의 부분
#######################
class NeuralSignalDataset(Dataset):
    def __init__(self, hdf_file_path):
        # HDF5 파일 로드
        with h5py.File(hdf_file_path, 'r') as f:
            # 예: 'signals' -> (N, C, T), 'labels' -> (N,) 형태 가정
            self.signals = np.array(f['signals'])
            self.labels = np.array(f['labels'])
        
        # 전처리 예: 정규화
        self.signals = (self.signals - np.mean(self.signals)) / (np.std(self.signals) + 1e-8)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        x = self.signals[idx]  # shape: (C, T)
        y = self.labels[idx]
        # Tensor 변환
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

#######################
# 2. 모델 정의 (간단한 예)
#######################
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch, C, T)
        # 간단히 모든 채널/시간 축을 Flatten
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#######################
# 3. Streamlit을 통한 데이터 시각화 (필요시 별도 실행)
#######################
def visualize_data(hdf_file_path):
    # streamlit run 이 스크립트를 별도로 실행하면 데이터 확인 가능
    with h5py.File(hdf_file_path, 'r') as f:
        signals = f['signals'][:100]  # 예: 앞 100개 신호
    st.title("Neural Signal Visualization")
    st.write("First sample signal")
    fig, ax = plt.subplots()
    ax.plot(signals[0].T)
    st.pyplot(fig)

#######################
# 4. 학습/평가 루프 정의
#######################
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

#######################
# 5. 메인 실행 (학습 파이프라인)
#######################
if __name__ == "__main__":
    # 가상의 HDF5 데이터 경로
    hdf_file_path = "neural_signals.h5"
    
    # 데이터셋 및 분할
    full_dataset = NeuralSignalDataset(hdf_file_path)
    train_size = int(len(full_dataset)*0.7)
    val_size = int(len(full_dataset)*0.15)
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델, 손실함수, 옵티마이저 정의
    # 여기서는 (C, T) -> Flatten 형태, input_dim = C*T 가정
    # 실제 값은 데이터 형태에 맞게 조정
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input[0].numel()
    hidden_dim = 128
    num_classes = len(torch.unique(full_dataset.labels))
    model = SimpleMLP(input_dim, hidden_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # MLflow 실험 관리 시작
    mlflow.set_experiment("NeuralSignal_Experiment")
    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("hidden_dim", hidden_dim)

        epochs = 10
        for epoch in range(1, epochs+1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # 메트릭 로깅
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 테스트 평가
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        
        # 모델 artifact 저장
        mlflow.pytorch.log_model(model, "model")

    # streamlit을 통한 데이터 시각화는 별도로 다음 명령으로 실행 가능
    # streamlit run train.py -- --visualize
```

### MLflow Tracking 사용법
- 위 코드를 실행하기 전에 `mlflow ui` 명령을 통해 MLflow UI를 띄울 수 있습니다.
- 모델 학습이 끝나면 `mlflow ui`로 웹 브라우저(기본 http://127.0.0.1:5000)에서 실험 결과, 메트릭, 파라미터 및 모델 아티팩트를 확인할 수 있습니다.

### Streamlit 사용 예시
- `streamlit run train.py -- --visualize` 와 같이 실행하여, `visualize_data()` 함수를 통해 HDF5 데이터 일부 샘플을 시각화할 수 있도록 할 수 있습니다.
- 실제로는 argparse 등을 사용하여 `--visualize` 플래그에 따라 `visualize_data()` 함수만 실행하는 로직을 추가할 수 있습니다.

---

위 예시는 전체 파이프라인을 간략히 보여주는 개념적 예시이며, 실제 환경, 데이터 전처리, 모델 구조, 하이퍼파라미터, GPU 사용 최적화 등은 실제 프로젝트 요구사항에 따라 조정하시기 바랍니다.