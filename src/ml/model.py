import wandb
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

# TODO: Maybe save all models on cpu
def load_model_from_wandb(run):
    model_name = run.config['model']['name']
    model_path = f"{run.project}/model-{run.id}:best"
    model_artifact = wandb.Api().artifact(model_path)

    if model_name == 'TemporalFusionTransformer':
        return TemporalFusionTransformer.load_from_checkpoint(model_artifact.file())

    raise ValueError("Invalid model name")