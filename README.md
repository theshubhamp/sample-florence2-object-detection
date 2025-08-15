# Sample: Object Detection over a Video Stream using Microsoft's Florence-2 Model 
Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

This repository hosts a sample for Florence-2's Object Detection capabilities that:
- Uses OpenCV to read a Video Frames
- Runs each frame through Florence-2 to get bounding boxes
- Overlay bounding boxes on top of the original image  

Here's a sample of how it looks live:
![alt text](https://github.com/user-attachments/assets/7d3c2d91-523f-4f35-827b-a85ba95136fd)

# Run on a Video File
```shell
uv run main.py ~/path/to/file.TS
```
