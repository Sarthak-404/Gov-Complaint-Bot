from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv
import torch

app = Flask(__name__)
CORS(app)
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

urgency_prompt = ChatPromptTemplate.from_template(
    """
    You are a complaint assistant. Your task is to analyze the complaint and determine whether it is an emergency
    or not. Always give answer in 'YES' and 'NO' only.
    Complaint: {input}
    """
)

query_prompt = ChatPromptTemplate.from_template(
    """
    You are a complaint assistant. Your task is to categorize user complaints into the following departments:
    Healthcare Ministry, Police, Public Works Department (PWD), Food Quality Ministry, Cleaning and Welfare Ministry, or Traffic Department.
    Also analyze the state of the complaint to declare it as an emergency and if it needs to be, tell the user it is an emergency complaint.
    Keep it short, about 50 words.
    Based on the user's complaint, tell them which department it has been assigned to and respond with:
    'Your complaint is registered with "Department name" and will be attended to shortly.'
    Complaint: {input}
    """
)

def process_complaint(complaint):
    main_query = query_prompt.invoke({'input': complaint})
    response = llm.invoke(main_query)
    department = response.content
    
    urgency_query = urgency_prompt.invoke({'input': complaint})
    urgent = llm.invoke(urgency_query)
    urgent_content = urgent.content
    
    return department, urgent_content

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route('/',methods=['GET'])
def home():
    return "Welcome to complaint assistant"

@app.route('/complaint', methods=['POST'])
def handle_complaint():
    data = request.json
    complaint = data.get('complaint')
    
    if not complaint:
        return jsonify({"error": "Complaint text is required"}), 400
    
    department, urgent = process_complaint(complaint)
    return jsonify({
        "department": department,
        "urgent": urgent
    })

@app.route('/caption', methods=['POST'])
def handle_image_caption():
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400
    
    image_file = request.files['image']
    image_path = f"./tmp/{image_file.filename}"
    image_file.save(image_path)
    caption = generate_caption(image_path)
    os.remove(image_path)
    
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)
