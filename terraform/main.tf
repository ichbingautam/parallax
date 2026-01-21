terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Optional: Remote state storage (recommended for team use)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "parallax"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
