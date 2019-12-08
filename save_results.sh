#!/bin/bash
# Save results from training on aws s3 bucket

echo "Saving Model Weights"
aws s3 sync models/ s3://rubix-cube-results/models
echo "Saving Training Logs"
aws s3 sync logs/gradient_tape s3://rubix-cube-results/logs/gradient_tape
echo "Saving Training Configs"
aws s3 sync logs/config s3://rubix-cube-results/logs/config


