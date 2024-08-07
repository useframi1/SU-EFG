# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the API directory contents into the container at /app
COPY ModelAPI/ /app

# Copy the utilities and pickle_files directories into the container at /app
COPY utilities/ /app/utilities
COPY pickle_files/ /app/pickle_files

# Copy the .env file into the container
COPY .env /app

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
