# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /DeployML

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variable for the model filename
ENV MODEL_FILENAME=housing_regression.pkl

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "fastAPI_app:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]
    