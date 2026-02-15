# Terraform Infrastructure

This directory contains `Terraform` configurations for provisioning cloud infrastructure for training on Google Cloud Platform (GCP).

## Prerequisites

1. **Install Terraform**: `brew install terraform`
2. **Install Google Cloud SDK**: `brew install google-cloud-sdk`
3. **Authenticate**: `gcloud auth application-default login`
4. **Set your project**: `gcloud config set project YOUR_PROJECT_ID`

## Configuration

The infrastructure is configurable via input variables. You should create a `terraform.tfvars` file to set these values.

### Available Variables

| Variable | Description | Type | Default | Required |
| :--- | :--- | :--- | :--- | :--- |
| `project_id` | GCP Project ID | string | - | **Yes** |
| `region` | GCP region for resources | string | `"us-central1"` | No |
| `zone` | GCP zone for TPU VMs | string | `"us-central1-a"` | No |
| `environment` | Environment name (dev, staging, prod) | string | `"dev"` | No |
| `tpu_type` | TPU accelerator type | string | `"v3-8"` | No |
| `tpu_runtime_version` | TPU runtime version | string | `"tpu-ubuntu2204-base"` | No |
| `machine_type` | Machine type for TPU VM host | string | `"n1-standard-8"` | No |
| `disk_size_gb` | Boot disk size in GB | number | `100` | No |
| `preemptible` | Use preemptible/spot TPUs (cheaper) | bool | `true` | No |

## Usage

1. **Initialize Terraform**:

    ```bash
    cd terraform
    terraform init
    ```

2. **Configure Environment**:

    Copy the example configuration and edit it with your project details:

    ```bash
    cp terraform.tfvars.example terraform.tfvars
    # Edit terraform.tfvars with your project_id and preferences
    ```

3. **Preview Changes**:

    ```bash
    terraform plan
    ```

4. **Apply Infrastructure**:

    ```bash
    terraform apply
    ```

5. **Connect to TPU VM**:

    ```bash
    gcloud compute tpus tpu-vm ssh parallax-dev-tpu --zone=us-central1-a
    ```

6. **Destroy Resources**:

    ⚠️ **Crucial**: Always destroy resources when not training to avoid high costs.

    ```bash
    terraform destroy
    ```

## Resources Created

- **TPU VM**: A Cloud TPU VM (default `v3-8`) for distributed JAX training.
- **GCS Bucket**: A Google Cloud Storage bucket for checkpoints and data.
- **VPC Network**: An isolated VPC network for training resources.
- **Service Account**: A dedicated service account with minimal required permissions for the TPU VM.

## Cost Warning

⚠️ **TPU v3-8 costs ~$8/hour (on-demand) or ~$2.40/hour (preemptible).**

Before running `terraform apply`, ensure you understand the pricing and have billing alerts set up. Always destroy resources when your training session is complete.
