FROM 233532778289.dkr.ecr.us-west-2.amazonaws.com/coverity-assist:dev1
# app code is already laid down by your overlay process
COPY app /app
# expose metrics client
RUN pip install --no-cache-dir prometheus-client
