import os
import argparse
import boto
import boto.s3

import os.path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", type=str, help="local dir with all the files to upload")
parser.add_argument("--bucket_name", type=str, help="s3 bucket name  to upload to")
parser.add_argument("--remote_dir", type=str, help="s3 remote dir under bucket to upload to")
args = vars(parser.parse_args())

'''
output_path = os.path.join(args["input_dir"], "final_output")
videos = [ os.path.join(output_path, x) for x  in os.listdir(output_path) if x.endswith(".mp4") ]
transcripts = [ os.path.join(output_path, x) for x in os.listdir(output_path) if x.endswith(".json") ]

#s3 upload file by file
for video, transcript in zip(videos, transcripts):
    print(f"uploading {video} ...")
    upload_to_s3(video)
    print(f"uploading {transcript} ...")
    upload_to_s3(transcript)
    print("done")
'''

# Fill these in - you get them when you sign up for S3
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY", None)
AWS_ACCESS_KEY_SECRET = os.environ.get("AWS_SECRET_KEY", None)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Fill in info on data to upload
# destination bucket name
bucket_name = args["bucket_name"]
# source directory
sourceDir = args["local_dir"]
# destination directory name (on s3)
destDir = args["remote_dir"]

#max size in bytes before uploading in parts. between 1 and 5 GB recommended
MAX_SIZE = 20 * 1000 * 1000
#size of parts when uploading in parts
PART_SIZE = 6 * 1000 * 1000

conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET)

bucket = conn.create_bucket(bucket_name,
        location=boto.s3.connection.Location.DEFAULT)


uploadFileNames = []
for (sourceDir, dirname, filename) in os.walk(sourceDir):
    uploadFileNames.extend(filename)
    break

def percent_cb(complete, total):
    sys.stdout.write('.')
    sys.stdout.flush()

for filename in uploadFileNames:
    sourcepath = os.path.join(sourceDir + filename)
    destpath = os.path.join(destDir, filename)
    print('Uploading %s to Amazon S3 bucket %s' % \
           (sourcepath, bucket_name))

    filesize = os.path.getsize(sourcepath)
    if filesize > MAX_SIZE:
        print("multipart upload")
        mp = bucket.initiate_multipart_upload(destpath)
        fp = open(sourcepath,'rb')
        fp_num = 0
        while (fp.tell() < filesize):
            fp_num += 1
            print("uploading part %i" %fp_num)
            mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10, size=PART_SIZE)

        mp.complete_upload()

    else:
        print("singlepart upload")
        k = boto.s3.key.Key(bucket)
        k.key = destpath
        k.set_contents_from_filename(sourcepath,
                cb=percent_cb, num_cb=10)
