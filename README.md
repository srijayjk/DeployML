# DeployML



**How to deploy model with Flask API**

* flask_app.py defines a Flask route `/predict` that handles HTTP POST requests.
* `request.get_json(force=True)`: Parses the incoming request data as JSON. `force=True` ensures that the data is parsed even if the `Content-Type` header is not explicitly set to `application/json`.
* `features = np.array(data['features']).reshape(1, -1)`:

  * Extracts the `features` list from the JSON data.
  * Converts the list to a NumPy array.
  * Reshapes the array to have a single row (`1`) and an automatically determined number of columns (`-1`), which is essential for most Scikit-learn models.
* Logging information is added.
* It checks whether the request contains the 'features' key, and returns a 400 error if it does not.
* `prediction = model.predict(features)`: Uses the loaded machine learning model to make a prediction.



**How to Run**

1. **Place your model:** Put your trained machine learning model's pickle file in a folder named `model_dump` in the same directory as the Python file.
   1. **Run from the command line:
      ```
      python flask_app.py --model housing_regression.pkl
      ```

Replace `your_model_file.pkl` with the actual name of your model file.

2. **Send a prediction request:** Use a tool like `curl`, Postman, or the provided PowerShell example to send a POST request to `http://127.0.0.1:5000/predict` with the input features in JSON format.
   ```
   $body = @{ features = @(-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252)} | ConvertTo-Json
   Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -Body $body
   ```



**How to Dockerise**

**Build the Docker image:**

Open a terminal in the directory where your `Dockerfile` is located and run the following command:

**Bash**

```
docker build -t flask-ml-api --build-arg MODEL_NAME=your_model_file.pkl .
```

Replace `your_model_file.pkl` with the actual name of your model file.

* `docker build`: Builds a Docker image.
* `-t flask-ml-api`: Tags the image with the name `flask-ml-api`.
* `--build-arg MODEL_NAME=your_model_file.pkl`: passes the model name argument to the docker build process.
* `.`: Specifies the current directory as the build context.

**Run the Docker container:**

After the image is built, you can run a container from it:

**Bash**

```
docker run -p 5000:5000 flask-ml-api
```

`docker run`: Runs a Docker container.

`-p 5000:5000`: Maps port 5000 on your host machine to port 5000 in the container.

`flask-ml-api`: Specifies the name of the image to run.


**Access the API:**

You can now access your Flask API at `http://localhost:5000`.
