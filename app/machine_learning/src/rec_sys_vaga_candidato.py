from pathlib import Path
from typing import List
from torchrec.modules.embedding_configs import PoolingType
import pickle
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from sklearn.model_selection import train_test_split
from common.utils import get_project_root
from dotenv import find_dotenv, load_dotenv
from common.logging import configure_logging
from loguru import logger
from tqdm import tqdm  # Added tqdm import

load_dotenv(find_dotenv())

project_name = os.getenv("PROJECT_NAME", "")
project_root_dir = get_project_root(project_name=project_name)

app_name = os.getenv("APP_NAME", "")
app_root_dir = get_project_root(project_name=app_name)

configure_logging(project_name=project_name, log_to_file=True, log_level="DEBUG")


# Step 1: Load and preprocess the data
def load_and_preprocess_data(parquet_path: str):
    """Load the parquet file and preprocess it for TorchRec"""
    logger.info(f"Starting data loading and preprocessing from: {parquet_path}")
    print("Loading data from", parquet_path)
    jobs_candidates_matrix = pd.read_parquet(parquet_path)
    logger.debug(f"Loaded DataFrame shape: {jobs_candidates_matrix.shape}")

    # Convert job IDs (index) and candidate IDs (columns) to integers for embedding lookup
    job_id_mapping = {
        job_id: idx for idx, job_id in enumerate(jobs_candidates_matrix.index)
    }
    candidate_id_mapping = {
        candidate_id: idx
        for idx, candidate_id in enumerate(jobs_candidates_matrix.columns)
    }
    logger.debug(f"Created job ID mapping with {len(job_id_mapping)} jobs.")
    logger.debug(
        f"Created candidate ID mapping with {len(candidate_id_mapping)} candidates."
    )

    # Create reverse mappings to convert back to original IDs
    job_idx_to_id = {idx: job_id for job_id, idx in job_id_mapping.items()}
    candidate_idx_to_id = {
        idx: candidate_id for candidate_id, idx in candidate_id_mapping.items()
    }

    # Create a dataset of (job_id, candidate_id, interviewed) tuples
    dataset = []
    logger.debug("Starting creation of dataset from DataFrame.")
    # Added tqdm for dataset creation
    for job_id in tqdm(
        jobs_candidates_matrix.index, desc="Processing job interactions"
    ):
        for candidate_id in jobs_candidates_matrix.columns:
            value = jobs_candidates_matrix.loc[job_id, candidate_id]
            # Only include entries where value is either 1 or None
            if pd.notna(value) or value is None:
                interviewed = 1.0 if pd.notna(value) and value == 1 else 0.0
                dataset.append(
                    {
                        "job_idx": job_id_mapping[job_id],
                        "candidate_idx": candidate_id_mapping[candidate_id],
                        "job_id": job_id,
                        "candidate_id": candidate_id,
                        "interviewed": interviewed,
                    }
                )
    logger.info(f"Created dataset with {len(dataset)} interactions.")
    df = pd.DataFrame(dataset)

    # Split into train/test sets
    logger.debug("Splitting data into training and testing sets.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(
        f"Data loaded: {len(df)} interactions, {len(job_id_mapping)} jobs, {len(candidate_id_mapping)} candidates"
    )
    print(f"Training set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    logger.info(
        f"Data loading and preprocessing finished. Train size: {len(train_df)}, Test size: {len(test_df)}"
    )

    # ----------- NEW CODE: Save pickles -----------
    output_dir = app_root_dir / "resources"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_objects = {
        "train_df.pkl": train_df,
        "test_df.pkl": test_df,
        "job_id_mapping.pkl": job_id_mapping,
        "candidate_id_mapping.pkl": candidate_id_mapping,
        "job_idx_to_id.pkl": job_idx_to_id,
        "candidate_idx_to_id.pkl": candidate_idx_to_id,
    }

    for filename, obj in save_objects.items():
        file_path = output_dir / filename
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved {filename} to {file_path}")
    # ----------------------------------------------

    return (
        train_df,
        test_df,
        job_id_mapping,
        candidate_id_mapping,
        job_idx_to_id,
        candidate_idx_to_id,
    )


# Step 2: Create batch collator for TorchRec
class JobCandidateCollator:
    """Collator to prepare batches for TorchRec model"""

    def __call__(self, batch_data):
        logger.debug(f"JobCandidateCollator called with batch size: {len(batch_data)}")
        job_indices = torch.tensor(
            [item["job_idx"] for item in batch_data], dtype=torch.int64
        )
        candidate_indices = torch.tensor(
            [item["candidate_idx"] for item in batch_data], dtype=torch.int64
        )
        labels = torch.tensor(
            [item["interviewed"] for item in batch_data], dtype=torch.float32
        )

        # Create sparse features using KeyedJaggedTensor
        # Each ID is represented as a list of a single ID (dense -> sparse conversion)
        features = KeyedJaggedTensor(
            keys=["job_id", "candidate_id"],
            values=torch.cat([job_indices, candidate_indices]),
            lengths=torch.ones(len(job_indices) * 2, dtype=torch.int64),
            offsets=torch.arange(0, len(job_indices) * 2 + 1, dtype=torch.int64),
        )
        logger.debug(
            f"Collated features: {features.keys()}, labels shape: {labels.shape}"
        )
        return features, labels


# Step 3: Define the recommendation model using TorchRec
class JobCandidateRecommender(nn.Module):
    """TorchRec-based recommendation model for job-candidate matching"""

    def __init__(self, num_jobs: int, num_candidates: int, embedding_dim: int = 64):
        super().__init__()
        logger.info(
            f"Initializing JobCandidateRecommender model with num_jobs={num_jobs}, num_candidates={num_candidates}, embedding_dim={embedding_dim}"
        )

        # Define embedding tables
        self.embedding_bag_configs = [
            EmbeddingBagConfig(
                name="job_id",
                num_embeddings=num_jobs,
                embedding_dim=embedding_dim,
                feature_names=["job_id"],
                pooling=PoolingType.MEAN,
            ),
            EmbeddingBagConfig(
                name="candidate_id",
                num_embeddings=num_candidates,
                embedding_dim=embedding_dim,
                feature_names=["candidate_id"],
                pooling=PoolingType.MEAN,
            ),
        ]
        logger.debug(f"EmbeddingBagConfigs created: {self.embedding_bag_configs}")

        # Create embedding layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model will use device: {self.device}")
        self.embedding_bags = EmbeddingBagCollection(
            tables=self.embedding_bag_configs,
            device=self.device,
        )
        logger.debug("EmbeddingBagCollection created.")

        # Interaction layer (dot product in this case)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        logger.debug("Output layer created.")

    def forward(self, features: KeyedJaggedTensor):
        logger.debug(f"Model forward pass started with feature keys: {features.keys()}")
        # Get embeddings
        embeddings = self.embedding_bags(features)
        logger.debug(f"Embeddings generated: {embeddings.keys()}")

        # Extract job and candidate embeddings
        job_embeddings = embeddings["job_id"]
        candidate_embeddings = embeddings["candidate_id"]
        logger.debug(
            f"Job embeddings shape: {job_embeddings.shape}, Candidate embeddings shape: {candidate_embeddings.shape}"
        )

        # Concatenate embeddings
        concat_embeddings = torch.cat([job_embeddings, candidate_embeddings], dim=1)
        logger.debug(f"Concatenated embeddings shape: {concat_embeddings.shape}")

        # Compute recommendation score
        score = self.output_layer(concat_embeddings)
        logger.debug(f"Output score shape before squeeze: {score.shape}")

        return score.squeeze()


# Step 4: Create DataLoader
class JobCandidateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for job-candidate interactions"""

    def __init__(self, dataframe):
        self.data = dataframe.to_dict("records")
        logger.info(f"JobCandidateDataset initialized with {len(self.data)} records.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Step 5: Training function
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """Train the recommendation model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(
        f"Starting model training on {device} for {epochs} epochs with learning rate {lr}."
    )

    # Binary classification loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger.debug("Loss function (BCELoss) and Optimizer (Adam) initialized.")

    print(f"Training on {device} for {epochs} epochs")

    # Added tqdm for epochs
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        # Training
        model.train()
        total_loss = 0
        # Added tqdm for training batches
        train_iterator = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False
        )
        for batch_idx, (features, labels) in enumerate(train_iterator):
            logger.debug(
                f"Epoch {epoch+1}, Training Batch {batch_idx+1}/{len(train_loader)}"
            )
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_test_loss = 0
        logger.debug(f"Epoch {epoch+1} - Starting evaluation on test set.")
        # Added tqdm for testing batches
        test_iterator = tqdm(
            test_loader, desc=f"Epoch {epoch+1}/{epochs} Testing", leave=False
        )
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(test_iterator):
                logger.debug(
                    f"Epoch {epoch+1}, Testing Batch {batch_idx+1}/{len(test_loader)}"
                )
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                test_iterator.set_postfix(loss=loss.item())

        avg_test_loss = total_test_loss / len(test_loader)
        logger.info(f"Epoch {epoch+1} - Average Test Loss: {avg_test_loss:.4f}")

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )
    logger.info("Model training finished.")
    return model


# Step 6: Make recommendations
def recommend_candidates(
    model,
    job_id,
    top_n,
    job_id_mapping,
    candidate_id_mapping,
    job_idx_to_id,
    candidate_idx_to_id,
    existing_interviews=None,
):
    """Generate top N candidate recommendations for a specific job"""
    logger.info(
        f"Starting candidate recommendation for job_id: {job_id}, top_n: {top_n}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Ensure model is in evaluation mode
    model = model.to(device)  # Ensure model is on the correct device
    logger.debug(f"Recommendation will use device: {device}")

    # Get job index
    if job_id in job_id_mapping:
        job_idx = job_id_mapping[job_id]
        logger.debug(f"Job ID {job_id} mapped to index {job_idx}.")
    else:
        logger.warning(
            f"Job ID {job_id} not found in training data. Returning empty list."
        )
        print(f"Job ID {job_id} not found in training data")
        return []

    # Create prediction data for all candidates
    all_candidates_data = []
    num_total_candidates = len(candidate_id_mapping)
    logger.debug(
        f"Generating prediction data for all {num_total_candidates} candidates."
    )
    # Added tqdm for creating all_candidates_data
    for candidate_idx in tqdm(
        range(num_total_candidates), desc="Preparing candidate data for scoring"
    ):
        all_candidates_data.append(
            {
                "job_idx": job_idx,
                "candidate_idx": candidate_idx,
                "interviewed": 0.0,  # Dummy label, not used in prediction
            }
        )

    candidate_indices_for_prediction = [c["candidate_idx"] for c in all_candidates_data]

    # Filter out candidates who were already interviewed if provided
    candidates_to_predict_df = pd.DataFrame(all_candidates_data)
    if existing_interviews is not None:
        logger.debug(f"Filtering already interviewed candidates for job {job_id}.")
        interviewed_for_this_job = existing_interviews.get(job_id, [])
        if interviewed_for_this_job:
            interviewed_candidate_indices = {
                candidate_id_mapping[cid]
                for cid in interviewed_for_this_job
                if cid in candidate_id_mapping
            }
            logger.debug(
                f"Found {len(interviewed_candidate_indices)} already interviewed candidates for this job."
            )

            # Create a boolean Series for filtering
            is_not_interviewed = ~candidates_to_predict_df["candidate_idx"].isin(
                interviewed_candidate_indices
            )
            candidates_to_predict_df = candidates_to_predict_df[is_not_interviewed]
            logger.info(
                f"Number of candidates to score after filtering: {len(candidates_to_predict_df)}"
            )
        else:
            logger.debug(
                f"No existing interviews found for job {job_id} or job not in existing_interviews."
            )

    # No candidates left to recommend
    if candidates_to_predict_df.empty:
        logger.warning(
            f"No candidates left to recommend for job {job_id} after filtering."
        )
        return []

    logger.debug(f"Number of candidates to score: {len(candidates_to_predict_df)}")

    # Create dataset and dataloader
    dataset = JobCandidateDataset(
        candidates_to_predict_df
    )  # Use the filtered DataFrame

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        collate_fn=JobCandidateCollator(),
    )
    logger.debug("Created DataLoader for prediction.")

    # Generate predictions
    predictions = []
    logger.debug("Starting prediction generation.")
    with torch.no_grad():
        # Added tqdm for prediction generation
        for (
            features,
            _,  # Labels are returned by collator but not needed for prediction
        ) in tqdm(dataloader, desc="Generating predictions"):
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
    logger.info(f"Generated {len(predictions)} predictions.")

    if len(predictions) != len(candidates_to_predict_df):
        logger.error(
            f"Mismatch in number of predictions ({len(predictions)}) and scored candidates ({len(candidates_to_predict_df)}). This should not happen."
        )

    candidate_scores = []
    # Added tqdm for mapping predictions to candidate IDs (might be fast, but consistent)
    for i, row in enumerate(
        tqdm(
            candidates_to_predict_df.itertuples(),
            total=len(candidates_to_predict_df),
            desc="Mapping scores",
        )
    ):
        original_candidate_id = candidate_idx_to_id[row.candidate_idx]
        score = predictions[i]
        candidate_scores.append((original_candidate_id, score))

    logger.debug(f"Mapped {len(candidate_scores)} scores to candidate IDs.")

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidate_scores[:top_n]
    logger.info(f"Returning top {len(top_candidates)} candidates for job {job_id}.")
    return top_candidates


#
def all_resources_exist(resources_dir: Path, saved_data_resources: List[str]) -> bool:
    """Check if all saved data resource files exist in the given directory."""
    existing_files = {file.name for file in resources_dir.glob("*.pkl")}
    return all(resource in existing_files for resource in saved_data_resources)


def load_data_resources(resources_dir: Path):
    for file in sorted(resources_dir.glob("*.pkl"), key=lambda f: f.name):
        with open(file, "rb") as pkl_f:
            yield pickle.load(pkl_f)


# Main function to run the whole pipeline
def main(parquet_path, job_id_to_recommend=None, top_n=5):
    logger.info("Starting main pipeline execution.")
    logger.info(f"Input parquet_path: {parquet_path}")
    if job_id_to_recommend:
        logger.info(
            f"Recommendations will be generated for job_id: {job_id_to_recommend}, top_n: {top_n}"
        )

    # Step 1: Load and preprocess data
    logger.info("Initiating Step 1: Load and preprocess data.")
    resources_dir = app_root_dir / "resources"
    saved_data_resources: List[str] = [
        "train_df.pkl",
        "test_df.pkl",
        "job_id_mapping.pkl",
        "candidate_id_mapping.pkl",
        "job_idx_to_id.pkl",
        "candidate_idx_to_id.pkl",
    ]
    if all_resources_exist(resources_dir, saved_data_resources):
        (
            candidate_id_mapping,
            candidate_idx_to_id,
            job_id_mapping,
            job_idx_to_id,
            test_df,
            train_df,
        ) = load_data_resources(resources_dir)

    else:
        (
            train_df,
            test_df,
            job_id_mapping,
            candidate_id_mapping,
            job_idx_to_id,
            candidate_idx_to_id,
        ) = load_and_preprocess_data(parquet_path)

    logger.info("Finished Step 1: Load and preprocess data.")

    # Step 2: Create datasets and dataloaders
    logger.info("Initiating Step 2: Create datasets and dataloaders.")
    train_dataset = JobCandidateDataset(train_df)
    test_dataset = JobCandidateDataset(test_df)

    collator = JobCandidateCollator()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collator
    )
    logger.debug(
        f"Train DataLoader created with batch_size=64, shuffle=True. Num batches: {len(train_loader)}"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, collate_fn=collator
    )
    logger.debug(
        f"Test DataLoader created with batch_size=64. Num batches: {len(test_loader)}"
    )
    logger.info("Finished Step 2: Create datasets and dataloaders.")

    # Step 3: Initialize and train the model
    logger.info("Initiating Step 3: Initialize and train the model.")
    model = JobCandidateRecommender(
        num_jobs=len(job_id_mapping), num_candidates=len(candidate_id_mapping)
    )
    logger.info("Model initialized.")

    trained_model = train_model(model, train_loader, test_loader, epochs=10)
    logger.info("Finished Step 3: Model training complete.")

    # ----------- NEW CODE: Save trained model weights -----------
    output_dir = app_root_dir / "resources"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "trained_model.pt"
    torch.save(trained_model.state_dict(), model_path)

    logger.info(f"Trained model weights saved to {model_path}")
    # ------------------------------------------------------------

    # Step 4: Generate recommendations if a job_id is provided
    if job_id_to_recommend:
        logger.info(
            f"Initiating Step 4: Generate recommendations for job_id {job_id_to_recommend}."
        )
        # Create a dictionary of existing interviews from the original matrix
        logger.debug("Loading original matrix to extract existing interviews.")
        jobs_candidates_matrix = pd.read_parquet(parquet_path)
        existing_interviews = {}
        # Added tqdm for extracting existing interviews
        for j_id_orig in tqdm(
            jobs_candidates_matrix.index, desc="Extracting existing interviews"
        ):
            existing_interviews[j_id_orig] = [
                candidate_id_orig
                for candidate_id_orig in jobs_candidates_matrix.columns
                if pd.notna(jobs_candidates_matrix.loc[j_id_orig, candidate_id_orig])
                and jobs_candidates_matrix.loc[j_id_orig, candidate_id_orig] == 1
            ]
        logger.debug(
            f"Extracted existing interviews for {len(existing_interviews)} jobs."
        )

        recommendations = recommend_candidates(
            trained_model,
            job_id_to_recommend,
            top_n,
            job_id_mapping,
            candidate_id_mapping,
            job_idx_to_id,
            candidate_idx_to_id,
            existing_interviews,
        )
        logger.info("Finished Step 4: Recommendations generated.")

        print(
            f"\nTop {len(recommendations)} recommended candidates for job {job_id_to_recommend}:"
        )
        for candidate_id, score in recommendations:
            print(f"Candidate ID: {candidate_id}, Score: {score:.4f}")
            logger.info(
                f"Recommendation for {job_id_to_recommend}: Candidate ID: {candidate_id}, Score: {score:.4f}"
            )
    else:
        logger.info("No job_id_to_recommend provided. Skipping recommendation step.")

    logger.info("Main pipeline execution finished.")
    return (
        trained_model,
        job_id_mapping,
        candidate_id_mapping,
        job_idx_to_id,
        candidate_idx_to_id,
    )


# Example usage
if __name__ == "__main__":
    logger.info("Starting script execution from __main__.")
    parquet_path = project_root_dir / "data/interacao_vagas_candidatos.parquet"
    logger.info(f"Parquet file path set to: {parquet_path}")

    # For demonstration, let's assume the first job ID in the matrix is the one we want recommendations for
    # In a real application, you would pass the specific job ID you're interested in
    if parquet_path.exists():
        logger.debug(f"Loading DataFrame to get a sample_job_id from {parquet_path}")
        jobs_candidates_matrix = pd.read_parquet(parquet_path)
        if not jobs_candidates_matrix.empty:
            sample_job_id = jobs_candidates_matrix.index[0]
            logger.info(f"Sample job ID for recommendation: {sample_job_id}")

            (
                model,
                job_id_mapping,
                candidate_id_mapping,
                job_idx_to_id,
                candidate_idx_to_id,
            ) = main(parquet_path, job_id_to_recommend=sample_job_id, top_n=5)
            logger.info("Script execution finished successfully.")
        else:
            logger.error(
                f"Parquet file at {parquet_path} is empty. Cannot proceed with example."
            )
            print(f"Error: Parquet file {parquet_path} is empty.")
    else:
        logger.error(f"Parquet file not found at {parquet_path}. Cannot run example.")
        print(f"Error: Parquet file {parquet_path} not found.")
