"""Sequential simulated federated client training."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from fedhydra.config import FederatedConfig
from fedhydra.methods.aggregation import model_delta
from fedhydra.utils import cosine_learning_rate, worker_seed


@dataclass(slots=True)
class ClientResult:
    client_id: int
    sample_count: int
    update: dict[str, torch.Tensor]
    mean_loss: float


def train_client(
    client_id: int,
    global_model: torch.nn.Module,
    dataset: Dataset,
    config: FederatedConfig,
    device: torch.device,
    round_index: int,
    total_rounds: int,
    seed: int,
    num_workers: int,
) -> ClientResult:
    if len(dataset) == 0:
        raise ValueError(f"client {client_id} has no samples")
    # Local stochasticity must depend only on the client/round seed. Otherwise
    # server-side VGAE sampling would perturb dropout and augmentation streams
    # relative to FedAvg/FedProx controls.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=loader_generator,
    )
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    global_state = {
        name: value.detach().clone() for name, value in global_model.state_dict().items()
    }
    reference_parameters = {
        name: value.detach().clone() for name, value in global_model.named_parameters()
    }
    learning_rate = cosine_learning_rate(config.learning_rate, round_index, total_rounds)
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = config.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    batches = 0
    for _ in range(config.local_epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(local_model(images), labels)
                if config.proximal_mu > 0:
                    proximal = torch.zeros((), device=device)
                    for name, parameter in local_model.named_parameters():
                        proximal += (parameter - reference_parameters[name]).square().sum()
                    loss = loss + 0.5 * config.proximal_mu * proximal
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
    update = model_delta(local_model.state_dict(), global_state)
    del local_model
    return ClientResult(
        client_id=client_id,
        sample_count=len(dataset),
        update=update,
        mean_loss=total_loss / max(batches, 1),
    )
