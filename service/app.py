from flask import Flask, request, jsonify
from core.recommender import GroupPlaylistService

app = Flask(__name__)
recommender_service = GroupPlaylistService()

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
