from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch

app = Flask(__name__)

# Load the VQA model and processor
from transformers import ViltForQuestionAnswering, ViltProcessor
vqa_model = ViltForQuestionAnswering.from_pretrained("/mnt/data/vilt_model")
vqa_processor = ViltProcessor.from_pretrained("/mnt/data/vilt_model")

# Set VQA model to evaluation mode and send to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqa_model.to(device)
vqa_model.eval()  # Ensure model is in eval mode

# Load the image captioning model, feature extractor, and tokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
caption_model = VisionEncoderDecoderModel.from_pretrained("./vit-gpt2-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("./vit-gpt2-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("./vit-gpt2-captioning")

# Send captioning model to device
caption_model.to(device)

# Global variable to store the uploaded image (for demo purposes)
current_image = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    global current_image
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        # Open and ensure the image is in RGB mode
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400

    # Store the image for later VQA (demo use only)
    current_image = image

    # Prepare image for captioning
    pixel_values = caption_feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    # Generate caption
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({'caption': caption})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    global current_image
    if current_image is None:
        return jsonify({'error': 'No image available. Please upload an image first.'}), 400

    question = request.form.get('question', None)
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Prepare the image and question for VQA
    encoding = vqa_processor(current_image, question, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Forward pass through the VQA model
    outputs = vqa_model(**encoding)
    logits = outputs.logits
    answer_idx = logits.argmax(-1).item()

    # Convert the raw index to a human-readable answer using the model's id2label mapping if available
    if hasattr(vqa_model.config, "id2label"):
        answer = vqa_model.config.id2label.get(answer_idx, str(answer_idx))
    else:
        answer = str(answer_idx)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
