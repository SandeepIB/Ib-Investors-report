import os
import configparser
from flask import Flask, render_template, request, flash, jsonify
from flask_toastr import Toastr
from langchain.chat_models import ChatOpenAI
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper

app = Flask(__name__)

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

app.config['SECRET_KEY'] = config.get('AppConfig', 'SECRET_KEY', fallback='mysecret')
os.environ["OPENAI_API_KEY"] = config.get('AppConfig', 'OPENAI_API_KEY')

toastr = Toastr(app)

# Constants
TOKEN = "9e846c60741647a2da"
DIRECTORY_PATH = config.get('AppConfig', 'DIRECTORY_PATH')

# Construct and save the index when the app starts
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    # Define the role and content for the user message
    user_role = "user"
    user_content = "Context: You are InfoBeans Investors assistant. Please share details only from the custom data provided to you."

    system_role = "system"
    system_content = "Context: You are InfoBeans Investors assistant. Please share details only from the custom data provided to you."
    # Create the messages input with role-based personalization
    messages = [
        {"role": user_role, "content": user_content},
        {"role": system_role, "content": system_content},
        # You can add more messages with different roles as needed
    ]

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs, messages=messages))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk(os.path.join(directory_path, 'index.json'))

    return index

@app.route('/', methods=['POST', 'GET'])
def investor():
    message = request.form.get('message')
    if not message:
        flash('Please ask me something', 'warning')
        return render_template('basic.html')

    response = chatbot(message)
    if response:
        return render_template('basic.html', input=message, result=response)
    else:
        flash('API request failed.', 'danger')
        return render_template('basic.html')

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(DIRECTORY_PATH, 'index.json'))
    response = index.query(input_text, response_mode="compact")
    return response.response

@app.route('/api', methods=['POST'])
def api():
    token = request.headers.get('Authorization')
    if token != f"Bearer {TOKEN}":
        return jsonify({'error': 'invalid token'}), 401

    data = request.get_json()
    if data is None or 'input' not in data:
        return jsonify({'error': 'invalid request, missing input parameter'}), 400

    output = chatbot(data['input'])
    return jsonify({'output': output}), 200

if __name__ == '__main__':
    # Construct and save the index when the app starts
    index = construct_index(DIRECTORY_PATH)
    app.run()
