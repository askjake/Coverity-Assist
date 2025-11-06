#!/usr/bin/env python3
import os, time, json, botocore
import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
PROFILE_NAME = os.environ.get("BEDROCK_PROFILE_NAME", "coverity-assist-api")
# System-defined inference profile to copy from (Claude 3.5 Sonnet, us-west-2)
COPY_FROM_ARN = os.environ.get(
    "BEDROCK_COPY_FROM",
    "arn:aws:bedrock:us-west-2:233532778289:inference-profile/"
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
)
TAGS = [
    {"key": "Application", "value": "coverity-assist-api"},  # shows up in Cost Explorer (once activated)
]

def main():
    bedrock = boto3.client("bedrock", region_name=REGION)
    try:
        resp = bedrock.create_inference_profile(
            inferenceProfileName=PROFILE_NAME,
            description="Coverity Assist API profile tagged for cost tracking",
            modelSource={"copyFrom": COPY_FROM_ARN},
            tags=TAGS,
        )
        # Response fields are top-level:
        arn = resp["inferenceProfileArn"]
        status = resp.get("status", "CREATING")
        print(json.dumps({"createdArn": arn, "status": status}, indent=2))
    except botocore.exceptions.ClientError as e:
        # If the name already exists, surface the ARN and exit cleanly
        if e.response.get("Error", {}).get("Code") == "ConflictException":
            print("Profile already exists; fetching details…")
            arn = f"arn:aws:bedrock:{REGION}:{boto3.client('sts').get_caller_identity()['Account']}:inference-profile/{PROFILE_NAME}"
        else:
            raise
    else:
        # Poll until ACTIVE
        arn = arn  # from create response

    # Wait for ACTIVE
    while True:
        info = bedrock.get_inference_profile(inferenceProfileIdentifier=arn)
        st = info["status"]
        print(f"Status: {st}")
        if st in ("ACTIVE", "FAILED", "DELETING", "DELETED"):
            break
        time.sleep(4)

    print(json.dumps({"inferenceProfileArn": arn, "finalStatus": st}, indent=2))

if __name__ == "__main__":
    main()
