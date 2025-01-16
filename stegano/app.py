import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64
import uuid

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper Functions
def double_to_byte(arr):
    return np.uint8(np.round(np.clip(arr, 0, 255), 0))

def increment_abs(x):
    return x + 1 if x >= 0 else x - 1

def decrement_abs(x):
    if np.abs(x) <= 1:
        return 0
    else:
        return x - 1 if x >= 0 else x + 1

def abs_diff_coefs(transform, u1, v1, u2, v2):
    return abs(transform[u1, v1]) - abs(transform[u2, v2])

def valid_coefficients(transform, bit, threshold, u1, v1, u2, v2):
    difference = abs_diff_coefs(transform, u1, v1, u2, v2)
    if (bit == 0) and (difference > threshold):
        return True
    elif (bit == 1) and (difference < -threshold):
        return True
    else:
        return False

def change_coefficients(transform, bit, u1, v1, u2, v2):
    coefs = transform.copy()
    if bit == 0:
        coefs[u1, v1] = increment_abs(coefs[u1, v1])
        coefs[u2, v2] = decrement_abs(coefs[u2, v2])
    elif bit == 1:
        coefs[u1, v1] = decrement_abs(coefs[u1, v1])
        coefs[u2, v2] = increment_abs(coefs[u2, v2])
    return coefs

def embed_bit(block, bit, u1, v1, u2, v2, P):
    patch = block.copy()
    coefs = cv2.dct(np.float32(patch))
    while not valid_coefficients(coefs, bit, P, u1, v1, u2, v2):
        coefs = change_coefficients(coefs, bit, u1, v1, u2, v2)
        patch = double_to_byte(cv2.idct(coefs))
    return patch

def view_as_blocks(arr, block_shape):
    arr_shape = np.array(arr.shape)
    block_shape = np.array(block_shape)
    blocks_shape = (arr_shape // block_shape) * block_shape
    blocks = arr[:blocks_shape[0], :blocks_shape[1]].reshape(
        -1, block_shape[0], blocks_shape[1] // block_shape[0], block_shape[1]
    ).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1])
    return blocks.reshape(blocks_shape[0] // block_shape[0], blocks_shape[1] // block_shape[1], *block_shape)

def embed_message(orig, msg, u1, v1, u2, v2, P):
    changed = orig.copy()
    blue = changed[:, :, 2]
    blue_padded = np.pad(blue, ((0, 8 - blue.shape[0] % 8), (0, 8 - blue.shape[1] % 8)), mode='constant')
    blocks = view_as_blocks(blue_padded, block_shape=(8, 8))
    h = blocks.shape[1]
    binary_message = ''.join([format(ord(char), '08b') for char in msg])
    for index, bit in enumerate(binary_message):
        i = index // h
        j = index % h
        block = blocks[i, j]
        blue_padded[i*8: (i+1)*8, j*8: (j+1)*8] = embed_bit(block, int(bit), u1, v1, u2, v2, P)
    changed[:, :, 2] = blue_padded[:blue.shape[0], :blue.shape[1]]
    return changed

def retrieve_bit(block, u1, v1, u2, v2):
    transform = cv2.dct(np.float32(block))
    return 0 if abs_diff_coefs(transform, u1, v1, u2, v2) > 0 else 1

def retrieve_message(img, length, u1, v1, u2, v2):
    blue = img[:, :, 2]
    blue_padded = np.pad(blue, ((0, 8 - blue.shape[0] % 8), (0, 8 - blue.shape[1] % 8)), mode='constant')
    blocks = view_as_blocks(blue_padded, block_shape=(8, 8))
    h = blocks.shape[1]
    binary_message = [str(retrieve_bit(blocks[index // h, index % h], u1, v1, u2, v2)) for index in range(length * 8)]
    binary_message = ''.join(binary_message)
    message = ''.join([chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8)])
    return message

def dct_insert(image, message, key):
    return embed_message(image, message, 4, 5, 5, 4, 25)

def dct_extract(image, key, message_length):
    return retrieve_message(image, message_length, 4, 5, 5, 4)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/insert', methods=['POST'])
def insert():
    """
    Endpoint to insert a message into an image
    
    Expects:
    - image file
    - message to hide
    - secret key
    """
    try:
        # Validate inputs
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"}), 400
        
        data = request.files['image']
        message = request.form.get('message')
        key = request.form.get('key')
        
        # Validate required parameters
        if not message or not key:
            return jsonify({"status": "error", "message": "Missing message or key"}), 400
        
        # Read and process image
        image = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Generate unique filename
        original_filename = data.filename
        name, ext = os.path.splitext(original_filename)
        hidden_filename = f"{name}_hidden.png"
        hidden_filepath = os.path.join(app.config['UPLOAD_FOLDER'], hidden_filename)
        
        # Modify image
        result_image = dct_insert(image, message, int(key))
        
        # Save modified image
        cv2.imwrite(hidden_filepath, result_image)
        
        # Encode image to base64 for transmission
        _, buffer = cv2.imencode('.png', result_image)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "image": encoded_img,
            "filename": hidden_filename
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    """
    Endpoint to extract a hidden message from an image
    
    Expects:
    - image file
    - secret key
    - expected message length
    """
    try:
        # Validate inputs
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"}), 400
        
        data = request.files['image']
        key = request.form.get('key')
        message_length = request.form.get('message_length')
        
        # Validate required parameters
        if not key or not message_length:
            return jsonify({"status": "error", "message": "Missing key or message length"}), 400
        
        # Read and process image
        image = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        extracted_message = dct_extract(image, int(key), int(message_length))
        
        return jsonify({
            "status": "success",
            "message": extracted_message
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
