## How to run ollama container

1) https://hub.docker.com/r/ollama/ollama --> Follow this to install ollama Docker image
2) Following command to run ollama container
    - ```docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama```
3) Pull model image with: ```curl http://localhost:11434/api/pull -d '{"model": "llama3.2"}'```
    - This also ensure you can connect to the container
4) Run second container
    - ```docker run --network host --rm --name python-client ollama-app```
    - Be sure that both container are in the ```host``` docker network