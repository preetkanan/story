from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("gpt2_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'story': story})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
