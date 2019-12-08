#!/bin/bash
# Save results from training on aws s3 bucket

echo ""
echo "Saving Model Weights"
aws s3 sync models/ s3://rubix-cube-results/models
echo ""
echo "Saving Training Logs"
aws s3 sync logs/gradient_tape s3://rubix-cube-results/logs/gradient_tape
echo ""
echo "Saving Training Configs"
aws s3 sync logs/configs s3://rubix-cube-results/logs/configs


