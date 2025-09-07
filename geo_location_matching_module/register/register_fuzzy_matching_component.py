from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

# Load the component from YAML
component = load_component(source="../component.yml")

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="68de562b-5fd0-4416-96b6-7b4c21574bcb",
    resource_group_name="ccb-ds-dev",
    workspace_name="ccb-ds-dev-ml"
)

# Register the component
ml_client.components.create_or_update(component)
