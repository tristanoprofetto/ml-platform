import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads an input file to a designated storage bucket.

    Args:
        bucket_name: Name of the bucket to upload to.
        source_file_name: Path to the file to upload.
        destination_blob_name: Destination name for the uploaded file.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


if __name__ == '__main__':
    bucket_name = 'mlflow-tristano'
    source_file_name = './data/feedback.csv'
    destination_blob_name = 'feedback.csv'
    upload_blob(bucket_name, source_file_name, destination_blob_name)
