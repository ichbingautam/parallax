# -----------------------------------------------------------------------------
# GCS Bucket for Checkpoints and Data
# -----------------------------------------------------------------------------

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "training_data" {
  name          = "parallax-${var.environment}-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = var.environment == "dev" # Only allow force destroy in dev

  uniform_bucket_level_access = true

  versioning {
    enabled = var.environment != "dev"
  }

  lifecycle_rule {
    condition {
      age = 30 # Delete old checkpoints after 30 days
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    project     = "parallax"
  }
}

# -----------------------------------------------------------------------------
# VPC Network
# -----------------------------------------------------------------------------

resource "google_compute_network" "training_network" {
  name                    = "parallax-${var.environment}-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "training_subnet" {
  name          = "parallax-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.training_network.id

  private_ip_google_access = true # Allow access to GCS without external IP
}

# Firewall rule for SSH access
resource "google_compute_firewall" "allow_ssh" {
  name    = "parallax-${var.environment}-allow-ssh"
  network = google_compute_network.training_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"] # Restrict this in production!
  target_tags   = ["parallax-training"]
}

# Firewall rule for internal communication
resource "google_compute_firewall" "allow_internal" {
  name    = "parallax-${var.environment}-allow-internal"
  network = google_compute_network.training_network.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/24"]
}

# -----------------------------------------------------------------------------
# Service Account
# -----------------------------------------------------------------------------

resource "google_service_account" "training_sa" {
  account_id   = "parallax-${var.environment}-training"
  display_name = "Parallax Training Service Account"
}

# Grant storage access
resource "google_storage_bucket_iam_member" "training_sa_storage" {
  bucket = google_storage_bucket.training_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.training_sa.email}"
}

# Grant TPU access
resource "google_project_iam_member" "training_sa_tpu" {
  project = var.project_id
  role    = "roles/tpu.admin"
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

# -----------------------------------------------------------------------------
# TPU VM
# -----------------------------------------------------------------------------

resource "google_tpu_v2_vm" "training_tpu" {
  name = "parallax-${var.environment}-tpu"
  zone = var.zone

  runtime_version  = var.tpu_runtime_version
  accelerator_type = var.tpu_type

  network_config {
    network           = google_compute_network.training_network.id
    subnetwork        = google_compute_subnetwork.training_subnet.id
    enable_external_ips = true
  }

  service_account {
    email  = google_service_account.training_sa.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  scheduling_config {
    preemptible = var.preemptible
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      set -e

      # Install Python and dependencies
      apt-get update
      apt-get install -y python3-pip python3-venv git

      # Clone and setup project
      cd /home
      git clone https://github.com/ichbingautam/parallax.git || true
      cd parallax

      # Create virtual environment
      python3 -m venv .venv
      source .venv/bin/activate

      # Install JAX for TPU
      pip install --upgrade pip
      pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install -e ".[dev]"

      echo "Setup complete! Run: source /home/parallax/.venv/bin/activate"
    EOF
  }

  labels = {
    environment = var.environment
    project     = "parallax"
  }
}
