variable "aws_region" {
  default = "eu-central-1"
}

variable "app_name" {
  default = "bayesian-model-api"
}

variable "container_image" {
  description = "ECR image URI"
  default     = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/bayesian-model:latest"
}

variable "container_port" {
  default = 8000
}