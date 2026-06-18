FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY app.py .

# Lambda handler
CMD ["app.handler"]