from PIL import Image, ImageDraw, ImageFont
import gradio as gr
# install timm pip install timm..
from transformers import pipeline
import scipy.io.wavfile as wavfile
# object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
# Use a pipeline as a high-level helper


# narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
narrator = pipeline("text-to-speech",
                    use_fast=False,
                    model="../models/models--kakao-enterprise--vits-ljs/snapshots/3bcb8321394f671bd948ebf0d086d694dda95464")
object_detector = pipeline("object-detection",
                           model="../models/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b")


def draw_bounding_boxes(image, detections, font_path=None):
    """
    Draw bounding boxes and labels on an image.

    Args:
        image (PIL.Image.Image): The input image.
        detections (list): List of detected objects. Each detection is a dictionary with:
            - "box" (tuple/list/dict): Bounding box as (xmin, ymin, xmax, ymax) or {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}.
            - "label" (str): Label of the detected object.
            - "confidence" (float, optional): Confidence score of the detection (0 to 1).
        font_path (str, optional): Path to a TTF font file for label text. Default uses default PIL font.

    Returns:
        PIL.Image.Image: The image with bounding boxes and labels drawn.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()

    for detection in detections:
        box = detection.get("box")
        if isinstance(box, dict):
            # Handle box as a dictionary
            xmin = box.get("xmin", 0)
            ymin = box.get("ymin", 0)
            xmax = box.get("xmax", 0)
            ymax = box.get("ymax", 0)
            box_coords = [(xmin, ymin), (xmax, ymax)]
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            # Handle box as a list or tuple
            box_coords = [(box[0], box[1]), (box[2], box[3])]
        else:
            # Skip invalid box format
            print(f"Invalid box format: {box}")
            continue

        label = detection.get("label", "Object")
        confidence = detection.get("confidence", None)

        # Draw the bounding box
        draw.rectangle(box_coords, outline="red", width=3)

        # Create the label text
        label_text = f"{label}"
        if confidence is not None:
            label_text += f" ({confidence:.2f})"

        # Determine text size and position
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (box_coords[0][0], box_coords[0][1] - text_height)

        # Draw the text background
        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill="red"
        )

        # Draw the label text
        draw.text(text_position, label_text, fill="white", font=font)

    return image




def generate_audio(text):
    narrated_text = narrator(text)
    print(narrated_text)
    wavfile.write("finetuned_output.wav", rate=narrated_text['sampling_rate'],
                  data=narrated_text["audio"][0])
    return "finetuned_output.wav"

def generate_human_readable_description(detections):
    """
    Generate a natural language description from detected objects.

    Args:
        detections (list): List of detected objects. Each detection is a dictionary with:
            - "box" (tuple/list/dict): Bounding box as (xmin, ymin, xmax, ymax) or {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}.
            - "label" (str): Label of the detected object.
            - "confidence" (float, optional): Confidence score of the detection (0 to 1).

    Returns:
        str: A grammatically correct, human-readable description of the detected objects.
    """
    if not detections:
        return "No objects were detected in the image."

    # Group detections by label
    label_counts = {}
    for detection in detections:
        label = detection.get("label", "object")
        confidence = detection.get("confidence", None)
        if label in label_counts:
            label_counts[label]["count"] += 1
            label_counts[label]["confidences"].append(confidence)
        else:
            label_counts[label] = {"count": 1, "confidences": [confidence]}

    # Build a grammatically correct description
    descriptions = []
    for label, info in label_counts.items():
        count = info["count"]
        confidences = [conf for conf in info["confidences"] if conf is not None]

        if count == 1:
            if confidences:
                descriptions.append(f"a {label} with a confidence of {confidences[0]:.2f}")
            else:
                descriptions.append(f"a {label}")
        else:
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            if avg_confidence:
                descriptions.append(f"{count} {label}s with an average confidence of {avg_confidence:.2f}")
            else:
                descriptions.append(f"{count} {label}s")

    # Combine descriptions into a grammatically correct sentence
    if len(descriptions) == 1:
        return f"The image contains {descriptions[0]}."
    else:
        return f"The image contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}."



def detect_image_and_draw_boundaries(image):
    output_pil = object_detector(image)
    processed_image = draw_bounding_boxes(image, output_pil)
    narrated_text=generate_human_readable_description(output_pil)
    processed_audio=generate_audio(narrated_text)
    return processed_image,processed_audio,narrated_text

gr_interface = gr.Interface(
    fn=detect_image_and_draw_boundaries,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Image(type="pil", label="Image with Bounding Boxes"),gr.Audio(label="Generated Audio"),gr.Text(label="narrated text")],
    title="Object Detection Viewer",
    description="Upload an image, and the app will draw bounding boxes and labels on it based on sample detections."
)

gr_interface.launch()
