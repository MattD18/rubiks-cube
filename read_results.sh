#!/bin/bash
# Read results from aws s3 bucket

echo "Reading Model Weights"
aws s3 cp s3://rubix-cube-results/models/ ./models/ --recursive
echo ""
echo "Reading Training Logs"
aws s3 cp s3://rubix-cube-results/logs/gradeint_tape ./logs/ec2 --recursive
echo ""
echo "Reading Training Configs"
aws s3 cp s3://rubix-cube-results/logs/configs ./logs/configs --recursive
