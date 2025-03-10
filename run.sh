# path="https://www.youtube.com/watch?v=pdQShdZaih8"
# path="data/images/InspireLab.jpeg"

# python multimodal_run.py --img_path $path
# python chatbot_run.py "This is an image of some members from Inspire Lab AI Team. Could you please provide me some information about them?"
#This is an image of some members from Inspire Lab AI Team. Could you please provide me some information about them?
#!/bin/bash

# Function to display help
function usage() {
    echo "Usage: $0 --file_path <path_to_image/video/pdf> --prompt <prompt_text> --enabled <input_type>"
    exit 1
}

# Parse the command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt) prompt="$2"; shift ;;
        --file_path) file_path="$2"; shift ;;
        --enabled) enabled="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if the required arguments are set
if [ -n "$file_path" ] && [ -n "$prompt" ]; then
    export PYTHONPATH=$(pwd)
    python scripts/multimodal_run.py --file_path "$file_path" --enabled "$enabled"
    python scripts/chatbot_run.py "$prompt"
elif [ -n "$prompt" ]; then
    export PYTHONPATH=$(pwd)
    python scripts/chatbot_run.py "$prompt"
else
    echo "Error: Missing required arguments."
    usage
fi