# Use the recommended base image langchain/langgraph-api:3.11
FROM langchain/langgraph-api:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project code to the container
COPY . /app

# Install any additional dependencies from your requirements.txt
RUN pip install --no-cache-dir -r my_agent/requirements.txt

# Expose the port that LangGraph server will run on
EXPOSE 8000

# Command to start the FastAPI server (and LangGraph if needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
