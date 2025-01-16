import logging
import sys
from flask import Flask, request, jsonify
from service.core.recommender import GroupPlaylistService

logging.getLogger('werkzeug').disabled = True
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *_: None

app = Flask(__name__)
app.logger.disabled = True
recommender_service = GroupPlaylistService('./data_files/train_sessions.jsonl', './model_files/wmf_model.pth', './log_files')

@app.route('/group_playlist', methods=['POST'])
def recommend():
    data = request.get_json()
    user_ids = data.get('user_ids', [])
    n = data.get('n', 30)
    
    try:
        recommendations = recommender_service.get_recommendations(
            user_ids=user_ids,
            n=n
        )
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
