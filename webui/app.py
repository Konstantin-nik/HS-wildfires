
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)
        return render_template('index.html', result="File Loaded")

if __name__ == '__main__':
    app.run(debug=True, port=5002)
