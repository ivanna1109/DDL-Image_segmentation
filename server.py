import flwr as fl
from typing import Dict, Optional, Tuple
from u_net_model import UNet
from preprocess_data import ImageMaskDataset, read_data
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from dice_b_loss import DiceBCELoss
import torch.optim as optim
from sklearn.model_selection import train_test_split

results_list = []

def get_eval_fn(model, test_loader):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        # Evaluate on the test dataset
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, masks in test_loader:
                outputs = model(images)
                loss += criterion(outputs, masks).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(masks.view_as(pred)).sum().item()
                total += masks.size(0)

        loss /= len(test_loader.dataset)
        accuracy = correct / total

        print(f"After round {server_round}, Global accuracy = {accuracy:.4f}")
        results = {"round": server_round, "loss": loss, "accuracy": accuracy}
        results_list.append(results)
        return loss, {"accuracy": accuracy}

    return evaluate

images, masks = read_data()
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageMaskDataset(train_images, train_masks, transform=transform)
test_dataset = ImageMaskDataset(test_images, test_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = UNet(in_channels=1, out_channels=1)
strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model, test_loader))
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )