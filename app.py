from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# --------------------
# Model definition
# --------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # SAFE reshape
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --------------------
# Load model
# --------------------
model = Net()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# --------------------
# Health check
# --------------------
@app.route("/", methods=["GET"])
def home():
    return "✅ Model API is running"

# --------------------
# Predict endpoint (SAFE)
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data or "input" not in data:
            return jsonify({
                "error": "Missing 'input'. Must be list of 784 floats"
            }), 400

        x = torch.tensor(data["input"], dtype=torch.float32)

        # Ensure correct shape
        if x.numel() != 28 * 28:
            return jsonify({
                "error": f"Expected 784 values, got {x.numel()}"
            }), 400

        x = x.view(1, 28 * 28)

        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()

        return jsonify({"prediction": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
