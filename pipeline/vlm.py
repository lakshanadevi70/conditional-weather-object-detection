
from collections import Counter

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Float16 is a practical default for CUDA GPUs such as the T4.
# Use float32 on CPU.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading VLM...")
print("Device:", DEVICE)
print("Dtype:", DTYPE)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
)

model = model.to(DEVICE)
model.eval()

print("VLM loaded successfully.")


def summarize_detections(detections):
    """
    Expected detection format:
    [
        {"class": "car", "confidence": 0.92,
         "bbox": [x1, y1, x2, y2]},
        {"class": "truck", "confidence": 0.88,
         "bbox": [x1, y1, x2, y2]}
    ]

    Counts are calculated from YOLO detections, not guessed by the VLM.
    """
    if not detections:
        return {
            "counts": {},
            "total": 0,
            "average_confidence": None,
            "summary": "No objects detected by YOLO."
        }

    counts = Counter()
    confidences = []

    for detection in detections:
        if isinstance(detection, dict):
            name = str(detection.get("class", "unknown"))
            confidence = detection.get("confidence")

        elif isinstance(detection, str):
            # Supports your previous list of class-name strings.
            name = detection
            confidence = None

        else:
            continue

        counts[name] += 1

        if confidence is not None:
            confidences.append(float(confidence))

    total = sum(counts.values())

    summary = "\n".join(
        f"- {name}: {count}"
        for name, count in sorted(counts.items())
    )

    if not summary:
        summary = "No valid detections were provided."

    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences else None
    )

    return {
        "counts": dict(counts),
        "total": total,
        "average_confidence": avg_conf,
        "summary": summary,
    }


def generate_explanation(
    image_path,
    condition="Unknown",
    detections=None,
    condition_confidence=None,
    question=None,
):
    """
    Generate an explanation or answer a question about an image.

    Parameters:
        image_path: image file path
        condition: classifier-predicted condition
        detections: list of YOLO detection dictionaries
        condition_confidence: classifier confidence (0 to 1)
        question: optional question from the user
    """
    image = Image.open(image_path).convert("RGB")

    detections = detections or []
    report = summarize_detections(detections)

    if condition_confidence is None:
        condition_text = str(condition)
    else:
        condition_text = (
            f"{condition} "
            f"({condition_confidence:.1%} confidence)"
        )

    avg_conf = report["average_confidence"]
    if avg_conf is None:
        confidence_text = "Not available"
    else:
        confidence_text = f"{avg_conf:.1%}"

    user_question = question or (
        "Explain the environmental condition and detected objects."
    )

    prompt = f"""
You are the explanation assistant for Smart Vision,
a condition-aware object detection system.

ENVIRONMENT CLASSIFIER RESULT:
{condition_text}

YOLO DETECTION SUMMARY:
{report["summary"]}

TOTAL DETECTED OBJECTS:
{report["total"]}

AVERAGE YOLO DETECTION CONFIDENCE:
{confidence_text}

RULES:
1. Use the provided classifier result as the predicted condition.
2. Use YOLO's detection summary for object names and counts.
3. Never invent objects, counts, confidence values, or measurements.
4. Explain that zero detections does not prove the image
   contains no objects.
5. Distinguish classifier confidence from detection confidence.
6. Explain how the condition may affect visibility when relevant.
7. If asked how object detection works, explain the pipeline:
   image -> condition classification -> optional preprocessing
   -> ESRGAN -> YOLO detection -> VLM explanation.
8. Do not claim that the VLM itself performs YOLO detection.
9. Give concise, clear, professional answers.
10. If information is unavailable, say so.

USER QUESTION:
{user_question}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    # Decode only the newly generated answer.
    answer_ids = generated_ids[:, input_length:]

    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
    )[0].strip()

    return answer


def answer_question(
    image_path,
    question,
    condition="Unknown",
    detections=None,
    condition_confidence=None,
):
    """Convenience function for interactive image questions."""
    return generate_explanation(
        image_path=image_path,
        condition=condition,
        detections=detections,
        condition_confidence=condition_confidence,
        question=question,
    )
