import os
from flask import Flask, render_template, request, jsonify
import ollama
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page for image upload and description."""
    return render_template('index.html')

@app.route('/describe', methods=['POST'])
def describe_image():
    """Handle image description using LLaVA."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Use Ollama to describe the image
            res = ollama.chat(
                model='llava:7b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in detail',
                        'images': [filepath]
                    }
                ]
            )
            
            # Return the description
            description = res['message']['content']
            return jsonify({
                'description': description,
                'image_path': f'/static/uploads/{filename}'
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)