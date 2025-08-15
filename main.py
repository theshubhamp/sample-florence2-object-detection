import sys
from queue import Queue
from threading import Thread

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Fallback to CPU (slow). Use cuda (nvidia) / mps (apple metal) if available.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# Frame Queues
frame_queue = Queue()
bboxes_queue = Queue()

def task_object_detection():
    prompt = "<OD>"
    while True:
        bboxes = []

        frame = frame_queue.get()
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(color_converted)

        inputs = processor(text=prompt, images=pil_frame, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(pil_frame.width, pil_frame.height))

        for box_index in range(len(parsed_answer["<OD>"]["bboxes"])):
            (x0, y0, x1, y1) = parsed_answer["<OD>"]["bboxes"][box_index]
            label = parsed_answer["<OD>"]["labels"][box_index]
            bboxes.append(((int(x0), int(y0)), (int(x1), int(y1)), label))

        bboxes_queue.put(bboxes)

def draw_bboxes(frame, bboxes):
    for ((x0, y0), (x1, y1), label) in bboxes:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x0), int(y0) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    desired_fps = 30 # TODO: Remove hardcoding / pick from video stream.
    delay_ms = int(1000 / desired_fps)  # Calculate delay in milliseconds

    video_path = sys.argv[1] # TODO: use argparse.
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("could not open video")
        exit()

    # run video / frame playback & inference / object detection concurrently (inference is slower than delay between frames)
    t = Thread(target=task_object_detection)
    t.daemon = True
    t.start()

    frame_count = 0
    old_bboxes = []

    while True:
        # Wait for a key press (e.g., 'q' to quit)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

        ret, frame = video.read()
        if not ret:
            break  # Break the loop if the video ends
        frame_count += 1

        # Enqueue first frame.
        if frame_count == 1:
            frame_queue.put(frame)

        try:
            # get bounding box for last submitted frame
            # note current frame != last submitted frame - so bounding can lag if inference in slow. OK, since the focus is on realtime video playback
            bboxes = bboxes_queue.get_nowait()
            old_bboxes = bboxes

            # enqueue current frame if the last frame we ran inference on finished.
            frame_queue.put(frame)
        except:
            pass

        # Draw bounding boxes based on object detection (last know bounding boxes, or updated ones from the try block above)
        draw_bboxes(frame, old_bboxes)
        cv2.imshow('Object Detection Output', frame)

    video.release()

if __name__ == '__main__':
    main()