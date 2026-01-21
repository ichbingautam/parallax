# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "bucket_name" {
  description = "GCS bucket name for training data and checkpoints"
  value       = google_storage_bucket.training_data.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.training_data.name}"
}

output "tpu_name" {
  description = "TPU VM name"
  value       = google_tpu_v2_vm.training_tpu.name
}

output "tpu_zone" {
  description = "TPU VM zone"
  value       = var.zone
}

output "service_account_email" {
  description = "Training service account email"
  value       = google_service_account.training_sa.email
}

output "ssh_command" {
  description = "SSH command to connect to TPU VM"
  value       = "gcloud compute tpus tpu-vm ssh ${google_tpu_v2_vm.training_tpu.name} --zone=${var.zone} --project=${var.project_id}"
}

output "training_command" {
  description = "Command to run training on TPU"
  value       = <<-EOF
    # SSH into TPU VM
    gcloud compute tpus tpu-vm ssh ${google_tpu_v2_vm.training_tpu.name} --zone=${var.zone}

    # Then run:
    cd /home/parallax
    source .venv/bin/activate
    python scripts/train.py --mode distributed --dataset tiny_shakespeare
  EOF
}
