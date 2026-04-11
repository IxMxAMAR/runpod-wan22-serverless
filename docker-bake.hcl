# Docker Bake configuration for WAN 2.2 serverless workers
# Build: docker buildx bake --file docker-bake.hcl
# Build specific target: docker buildx bake --file docker-bake.hcl t2v

variable "REGISTRY" {
  default = ""
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["base", "t2v", "i2v"]
}

target "base" {
  dockerfile = "docker/Dockerfile.base"
  tags       = ["${REGISTRY}wan22-base:${TAG}"]
  platforms  = ["linux/amd64"]
}

target "t2v" {
  dockerfile = "docker/Dockerfile.t2v"
  tags       = ["${REGISTRY}wan22-t2v:${TAG}"]
  platforms  = ["linux/amd64"]
  contexts = {
    "wan22-base:latest" = "target:base"
  }
}

target "i2v" {
  dockerfile = "docker/Dockerfile.i2v"
  tags       = ["${REGISTRY}wan22-i2v:${TAG}"]
  platforms  = ["linux/amd64"]
  contexts = {
    "wan22-base:latest" = "target:base"
  }
}
