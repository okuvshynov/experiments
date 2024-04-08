from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    # Echo back the received data
    received_data = request.get_json()
    response = {'echo': received_data}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)