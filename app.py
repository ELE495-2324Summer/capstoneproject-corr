from flask import Flask, request, jsonify

app = Flask(_name_)

number = 0
score = 0
plate_number = ""
park_status = "incomplete"

@app.route('/send_number', methods=['POST'])
def send_number():
    global number
    data = request.get_json()
    number = data.get('number', 0)
    return jsonify({"status": "success"}), 200

@app.route('/get_number', methods=['GET'])
def get_number():
    return jsonify({"number": number}), 200

@app.route('/plate', methods=['POST'])
def update_plate():
    global plate_number
    data = request.get_json()
    plate_number = data.get('plate', "")
    print(f"Received plate: {plate_number}")
    return jsonify({"status": "plate received"}), 200

@app.route('/get_plate', methods=['GET'])
def get_plate():
    return jsonify({"plate": plate_number}), 200

@app.route('/score', methods=['POST'])
def update_score():
    global score
    data = request.get_json()
    score = data.get('score', 0)
    return jsonify({"status": "score updated", "score": score}), 200

@app.route('/get_score', methods=['GET'])
def get_score():
    return jsonify({"score": score}), 200

@app.route('/park_complete', methods=['POST'])
def park_complete():
    global park_status
    data = request.get_json()
    park_status = data.get('status', "incomplete")
    return jsonify({"status": "park status updated", "park_status": park_status}), 200

@app.route('/get_park_complete', methods=['GET'])
def get_park_complete():
    return jsonify({"park_status": park_status}), 200

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "number": number,
        "score": score,
        "plate_number": plate_number,
        "park_status": park_status
    }), 200

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)
