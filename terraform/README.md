# Terraform Infrastructure

This directory contains Terraform configurations for provisioning cloud infrastructure for training.

## Prerequisites

1. Install Terraform: `brew install terraform`
2. Install Google Cloud SDK: `brew install google-cloud-sdk`
3. Authenticate: `gcloud auth application-default login`
4. Set your project: `gcloud config set project YOUR_PROJECT_ID`

## Usage

```bash
cd terraform

# Initialize Terraform
terraform init

# Preview changes
terraform plan -var="project_id=YOUR_PROJECT_ID"

# Apply infrastructure
terraform apply -var="project_id=YOUR_PROJECT_ID"

# Destroy when done (important to avoid costs!)
terraform destroy -var="project_id=YOUR_PROJECT_ID"
```

## Resources Created

- **TPU VM**: v3-8 TPU for distributed JAX training
- **GCS Bucket**: For checkpoints and data storage
- **VPC Network**: Isolated network for training
- **Service Account**: With minimal required permissions

## Cost Warning

⚠️ TPU v3-8 costs ~$8/hour. Always destroy resources when not training!
