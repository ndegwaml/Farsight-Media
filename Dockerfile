# Use the official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "bert.py", "--server.port=8501", "--server.enableCORS=false"]
