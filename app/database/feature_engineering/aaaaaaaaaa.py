from pathlib import Path
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    logging.info("Carregando arquivo Parquet: %s", file_path)
    df = pd.read_parquet(file_path)
    logging.info("DataFrame carregado com shape: %s", df.shape)
    return df


def convert_to_sparse(df: pd.DataFrame):
    logging.info("Convertendo DataFrame para matriz esparsa CSR")
    values = df.values
    sparse_matrix = csr_matrix(values)
    logging.info("Matriz esparsa CSR - formato: %s", sparse_matrix.shape)

    logging.info("Convertendo matriz esparsa para tensor esparso do PyTorch")
    coo = sparse_matrix.tocoo()
    indices = torch.LongTensor(np.ndarray([coo.row, coo.col]))
    values_t = torch.FloatTensor(coo.data)
    logging.info("Tensor esparso PyTorch - índices:\n%s", indices)
    logging.info("Tensor esparso PyTorch - valores:\n%s", values_t)

    logging.info("Convertendo DataFrame para tensor denso do PyTorch")
    dense_tensor = torch.FloatTensor(values)
    logging.info("Tensor denso PyTorch:\n%s", dense_tensor)

    return values, dense_tensor


def split_data(user_item_matrix, train_ratio=0.7, val_ratio=0.15):
    # Transpor para ter shape (num_usuarios, num_vagas)
    logging.info("Transpondo matriz para ter o formato (num_usuarios, num_vagas)")
    user_item_matrix = user_item_matrix.T
    num_users = user_item_matrix.shape[0]
    logging.info("Número de usuários: %d", num_users)

    indices = np.arange(num_users)
    np.random.shuffle(indices)
    train_idx = indices[: int(train_ratio * num_users)]
    val_idx = indices[
        int(train_ratio * num_users) : int((train_ratio + val_ratio) * num_users)
    ]
    test_idx = indices[int((train_ratio + val_ratio) * num_users) :]

    logging.info(
        "Índices de usuários - treino: %s, validação: %s, teste: %s",
        train_idx,
        val_idx,
        test_idx,
    )

    train_data = user_item_matrix[train_idx]
    val_data = user_item_matrix[val_idx]
    test_data = user_item_matrix[test_idx]
    return train_data, val_data, test_data


class AutoRecDataset(Dataset):
    def __init__(self, data_matrix):
        """
        data_matrix: numpy array ou lista de listas, shape (n_samples, n_features).
        Converte para tensor PyTorch.
        """
        self.data = torch.FloatTensor(data_matrix)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Retorna tupla (input, target) para o autoencoder
        x = self.data[idx]
        return x, x


class AutoRec(nn.Module):
    def __init__(self, num_items, hidden_dim):
        super(AutoRec, self).__init__()
        # Encoder: reduz de num_items para hidden_dim
        self.encoder = nn.Linear(num_items, hidden_dim)
        # Decoder: expande de hidden_dim de volta para num_items
        self.decoder = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        # x: tensor shape (batch_size, num_items)
        hidden = torch.sigmoid(self.encoder(x))
        out = self.decoder(hidden)
        return out


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    early_stop_patience=5,
    improvement_threshold=1e-4,
):
    logging.info("Iniciando treinamento por %d épocas", num_epochs)
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        logging.info("Época %d/%d, Loss: %.4f", epoch + 1, num_epochs, epoch_loss)

        if best_loss - epoch_loss > improvement_threshold:
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            logging.info(
                "Early stopping ativado na época %d devido à pouca melhoria consecutiva.",
                epoch + 1,
            )
            break


def evaluate(model, dataloader, device):
    model.eval()
    mse = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            mse += nn.MSELoss(reduction="sum")(outputs, targets).item()
    mse /= len(dataloader.dataset)
    rmse = np.sqrt(mse)
    logging.info("Avaliação RMSE: %.4f", rmse)
    return rmse


def save_model(model, output_path):
    logging.info("Salvando modelo em: %s", output_path)
    torch.save(model.state_dict(), output_path)
    logging.info("Modelo salvo com sucesso.")


def main():
    parquet_file = "temp.parquet"
    model_save_path = "autorec_model.pth"

    # Autoseleção do device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device selecionado: %s", device)

    # Carregar dados
    df = load_dataframe(parquet_file)

    # Conversão para matriz esparsa e tensor denso (opcionalmente usando o tensor esparso)
    values, _ = convert_to_sparse(df)

    # Dividir os dados
    train_data, val_data, test_data = split_data(values)

    # Criação dos datasets e dataloaders
    train_dataset = AutoRecDataset(train_data)
    val_dataset = AutoRecDataset(val_data)
    test_dataset = AutoRecDataset(test_data)

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instância do modelo e envio para o device selecionado
    num_items = train_data.shape[1]  # número de vagas
    hidden_dim = 2
    model = AutoRec(num_items=num_items, hidden_dim=hidden_dim).to(device)

    # Critério de perda e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Treinamento com early stopping
    logging.info("Iniciando treinamento do modelo AutoRec")
    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_epochs=50,
        early_stop_patience=5,
        improvement_threshold=1e-4,
    )

    # Avaliação
    train_rmse = evaluate(model, train_loader, device)
    val_rmse = evaluate(model, val_loader, device)
    test_rmse = evaluate(model, test_loader, device)
    logging.info("RMSE - Treino: %.4f", train_rmse)
    logging.info("RMSE - Validação: %.4f", val_rmse)
    logging.info("RMSE - Teste: %.4f", test_rmse)

    # Salvar o modelo treinado
    save_model(model, model_save_path)

    # Limpar a cache da CUDA, se estiver usando GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logging.info("Cache CUDA limpa.")


if __name__ == "__main__":
    main()
