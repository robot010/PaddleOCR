---
annotations_creators: [manual]
language: [en]
license: mit
multilinguality: [monolingual]
size_categories: [n<1K]
source_datasets: [original]
task_categories: [image-classification]
task_ids: [multi-class-image-classification]
pretty_name: Medical Prescription Handwritten Words
---

# Medical Prescription Handwritten Words

This dataset contains images of individual handwritten medical words extracted from prescription notes. It is designed for training and evaluating handwriting recognition models in the healthcare domain.

## Structure

- `images/`: Contains 40+ handwritten word images (e.g., `Amoxicillin.png`, `Cold.png`, `Tablet.png`, `0.png`, etc.)
- `data.csv`: Maps each image file to its corresponding label (word)

## Example Use Cases

- OCR (Optical Character Recognition) for medical prescriptions
- AI-powered digitization of handwritten medical records
- Deep learning models for handwriting recognition

## Labels

Some example words:
- Amoxicillin
- Pain
- Fever
- Tablet
- Twice
- Syrup
- Cold
- Pressure


Dataset by [avi-kai](https://huggingface.co/avi-kai)
