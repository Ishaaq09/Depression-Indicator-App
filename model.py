import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from preprocessing import preprocess_data, preprocessor
from sklearn.metrics import precision_score, recall_score, f1_score


class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.network(x)


def train_model(epochs=20, lr=0.001):
    train_loader, test_loader, input_dim, (X_test_tensor, y_test_tensor) = preprocess_data()

    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_classes = (predictions > 0.5).float()

    # Convert to numpy arrays
    y_true = y_test_tensor.numpy()
    y_pred = predicted_classes.numpy()

    accuracy = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Test Accuracy : {accuracy*100:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")

    # Save model
    torch.save(model.state_dict(), "depression_model.pth")
    print("Model saved as depression_model.pth")

    # Save preprocessor
    with open("preprocessor_pipeline.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved as preprocessor_pipeline.pkl")


if __name__ == "__main__":
    train_model()
