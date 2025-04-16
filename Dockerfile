FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
