import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total if total else 0.0
    accuracy = (correct / total) * 100 if total else 0.0

    return avg_loss, accuracy
