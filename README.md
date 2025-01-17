# Group Playlist Recommendation Microservice

## Project Description

This microservice is built using Python and Flask and provides group playlist recommendations based on users input.  
It features two recommendation models:

1.  A **basic model** that recommends songs by analyzing the most frequently listened-to tracks within a user group.
2.  An **advanced model** leveraging a pre-trained Weighted Matrix Factorization (WMF) algorithm.

Logs of each API request are stored in files within a Docker volume, enabling easy monitoring.

## Features

- API for generating playlists.
- Logging of all incoming requests to dedicated files for A/B experiment.
- Support for running the service in a Docker container.

---

## Requirements

- All dependencies are listed in `pyproject.toml`.

---

## Installation and Running the Service

### Running Locally

1. Clone the repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd <PROJECT_NAME>
   ```
2. Install dependencies:
   ```bash
   pip install --no-cache-dir .
   ```
3. Run the Flask server:
   ```bash
   PYTHONPATH=<PATH_TO_REPO> python service/app.py \
   	   --sessions-data <filepath> \
   	   --model-path <filepath> \
   	   --log-dir <path> \
   	   --port <port>
   ```
4. API will be available at http://127.0.0.1:5000

### Running with Docker

1. Clone the repository and navigate to project directory:
2. Build the docker image:
   ```bash
   docker build -t <image_name> .
   ```
3. Run the docker conatiner:
   ```bash
   docker run -d -p 5000:5000 \
   	   -v <log_directory_path>:/app/logs \
   	   --name recommendations \
   	   recommendations:latest \
   	   --sessions-data <filepath> \
   	   --model-path <filepath> \
   	   --log-dir <path> \
   	   --port <port>
   ```
4. API will be available at http://127.0.0.1:5000

## API Endpoints

### POST `/group_playlist`

Generates playlist for users based on the input data.

#### Sample Request:

```bash
{
  "user_ids": [123. 332. 45. 624. 123],
  "n": 30
}
```

#### Sample Response:

```bash
{
  "group_size": 8,
  "tracks": ["3aEJMh1cXKEjgh52claxQp", ... , "0Hsc0sIaxOxXBZbT3ms2oj" ]
}
```

## Logging

All API requests are logged in files under the `logs/` directory.  
When running the service in Docker, logs are stored on the host machine via a Docker volume.
