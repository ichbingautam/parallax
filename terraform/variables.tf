# -----------------------------------------------------------------------------
# Required Variables
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

# -----------------------------------------------------------------------------
# Optional Variables
# -----------------------------------------------------------------------------

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for TPU VMs"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "tpu_type" {
  description = "TPU accelerator type"
  type        = string
  default     = "v3-8" # 8 TPU v3 cores
}

variable "tpu_runtime_version" {
  description = "TPU runtime version"
  type        = string
  default     = "tpu-ubuntu2204-base"
}

variable "machine_type" {
  description = "Machine type for TPU VM host"
  type        = string
  default     = "n1-standard-8"
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

variable "preemptible" {
  description = "Use preemptible/spot TPUs (cheaper but can be interrupted)"
  type        = bool
  default     = true
}
