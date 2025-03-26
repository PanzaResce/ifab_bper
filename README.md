# ifab_bper

## To run ollama-app
1. Install and execute ollama https://ollama.com
2. For GPU acceleration:
```OLLAMA_ENABLE_CUDA=1```
3. If WSL, on Windows:
```$env:OLLAMA_HOST="0.0.0.0"; ollama serve```
Otherwise:
```OLLAMA_HOST=0.0.0.0 ollama serve```

4. Update the ```OLLAMA_URL``` variable with the correct endpoint for accessing the Ollama API.
5. ```sudo docker build -t my-ollama-app .```
6. ```sudo docker run --rm --name python-client my-ollama-app```