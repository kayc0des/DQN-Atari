# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables to avoid Python buffering
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application will run on
EXPOSE 5000

# Command to run the application (this can be adjusted depending on how you want to run it)
CMD ["python", "train.py"]
