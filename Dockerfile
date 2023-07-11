# Use an official Python runtime as the base image
FROM python:3.10.6-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt requirements.txt

# Copy the application code to the working directory
COPY .streamlit .streamlit
COPY app/app.py app/app.py
COPY app/game.py app/game.py
COPY registry.py registry.py
COPY data_proc.py data_proc.py


# Install the dependencies
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/0_👋_Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Define the command to run the application when the container starts
# CMD streamlit run app/app.py
