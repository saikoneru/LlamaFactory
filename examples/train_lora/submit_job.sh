#!/bin/bash
# Usage: ./submit_job.sh /path/to/your/config.yaml

CONFIG_PATH="$1"

if [ -z "$CONFIG_PATH" ]; then
    echo "❌ Error: Please provide the configuration path to the qwen 2 omni lora when folliwing the example."
    echo "Usage: ./submit_job.sh <path_to_yaml_config>"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config file '$CONFIG_PATH' does not exist!"
    exit 1
fi

echo "Submitting job with config: $CONFIG_PATH"
sbatch --export=CONFIG_PATH="$CONFIG_PATH" submit_bentxt.sbatch
