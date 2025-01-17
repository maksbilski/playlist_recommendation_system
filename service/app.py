import logging
import sys
import argparse
from flask import Flask, request, jsonify
from service.core.recommender import GroupPlaylistService

def parse_args():
    parser = argparse.ArgumentParser(description='Playlist Recommendation Service')
    parser.add_argument('--train-data', 
                      default='./data_files/train_sessions.jsonl',
                      help='Path to training data file')
    parser.add_argument('--model-path', 
                      default='./model_files/wmf_model.pth',
                      help='Path to model file')
    parser.add_argument('--log-dir', 
                      default='./log_files',
                      help='Directory for logs')
    parser.add_argument('--port',
                      type=int,
                      default=5000,
                      help='Port number')
    return parser.parse_args()

def create_app(train_data_path, model_path, log_dir):
    logging.getLogger('werkzeug').disabled = True
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *_: None
    
    app = Flask(__name__)
    app.logger.disabled = True
    
    app.recommender_service = GroupPlaylistService(
        train_data_path,
        model_path,
        log_dir
    )
    
    @app.route('/group_playlist', methods=['POST'])
    def recommend():
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        n = data.get('n', 30)
        
        try:
            recommendations = app.recommender_service.get_recommendations(
                user_ids=user_ids,
                n=n
            )
            return jsonify(recommendations)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return app

if __name__ == '__main__':
    args = parse_args()
    app = create_app(args.train_data, args.model_path, args.log_dir)
    app.run(host='0.0.0.0', port=args.port)
