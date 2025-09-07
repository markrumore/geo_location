import os
from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential

# Use environment variables for config, with sensible defaults
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_ML_WORKSPACE")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)

# Data asset names and versions can also be parameterized
customers_asset_name = os.getenv("CUSTOMERS_ASSET_NAME", "geo_location_matching_module-tests-data-test_customers-csv")
customers_asset_version = os.getenv("CUSTOMERS_ASSET_VERSION", "1")
unmatched_asset_name = os.getenv("UNMATCHED_ASSET_NAME", "geo_location_matching_module-tests-data-test_unmatched-csv")
unmatched_asset_version = os.getenv("UNMATCHED_ASSET_VERSION", "1")

customers_asset = ml_client.data.get(customers_asset_name, version=customers_asset_version)
unmatched_asset = ml_client.data.get(unmatched_asset_name, version=unmatched_asset_version)

def test_fuzzy_matching_pipeline():
    fuzzy_matching_component_name = os.getenv("FUZZY_MATCHING_COMPONENT_NAME", "fuzzy_matching_component")
    fuzzy_matching_component = ml_client.components.get(fuzzy_matching_component_name)

    @dsl.pipeline(
        compute=os.getenv("AZURE_COMPUTE_NAME", "shared-D13"),
        description="Test pipeline for fuzzy matching component"
    )
    def pipeline_func(
        input_customers,
        input_unmatched,
        keep_all=False,
        zip_col1="POSTAL_CODE",
        name_col1="CUSTOMER_DESC",
        address_col1="STREET_ADDRESS",
        lat_col1="LATITUDE_COORDINATE",
        long_col1="LONGITUDE_COORDINATE",
        zip_col2="POSTAL_CODE",
        name_col2="CUSTOMER_DESC",
        address_col2="STREET_ADDRESS_LINE_1",
        lat_col2="LATITUDE",
        long_col2="LONGITUDE",
        threshold=95,
        lat_long_tolerance=3
    ):
        fuzzy_step = fuzzy_matching_component(
            input_customers=input_customers,
            input_unmatched=input_unmatched,
            output_path="outputs/",
            keep_all=keep_all,
            zip_col1=zip_col1,
            name_col1=name_col1,
            address_col1=address_col1,
            lat_col1=lat_col1,
            long_col1=long_col1,
            zip_col2=zip_col2,
            name_col2=name_col2,
            address_col2=address_col2,
            lat_col2=lat_col2,
            long_col2=long_col2,
            threshold=threshold,
            lat_long_tolerance=lat_long_tolerance
        )
        return {
            "matched_results": fuzzy_step.outputs.matched_results
        }

    test_customers = Input(
        path=customers_asset.id,
        type=AssetTypes.URI_FILE,
        mode=InputOutputModes.RO_MOUNT
    )
    test_unmatched = Input(
        path=unmatched_asset.id,
        type=AssetTypes.URI_FILE,
        mode=InputOutputModes.RO_MOUNT
    )

    pipeline_job = pipeline_func(
        input_customers=test_customers,
        input_unmatched=test_unmatched
    )

    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline job submitted: {submitted_job.name}")

