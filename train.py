import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.config import hyperparams
from model.model import VertexDetectionModel
from utils.dataloader import loadData


def trainingLoop(hyperparams: dict):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = loadData(
        'sample_datasets/image',
        'sample_datasets/mask',
        0.8,
        **{'batch_size': hyperparams['batch_size']}
    )

    model = VertexDetectionModel().to(device)
    weight = torch.tensor([hyperparams['class_weight']] * hyperparams['batch_size'])
    criterion = nn.BCELoss(weight=weight,reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    for epoch in range(hyperparams['epochs']):
        model.train()
        running_loss = 0.0

        # Initialize tqdm progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{hyperparams['epochs']}")
            for images, masks in tepoch:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()

                pred = model(images)
                loss = criterion(pred, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update progress bar with the current loss
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{hyperparams['epochs']}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation (Optional)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                log_loss, one_minus_log_loss = model(images)
                loss = criterion(log_loss, masks) + criterion(one_minus_log_loss, masks)
                val_loss += loss.item()

            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    print("Training complete.")

if __name__ == '__main__':
    trainingLoop(hyperparams)
