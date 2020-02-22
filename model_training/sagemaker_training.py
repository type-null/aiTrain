from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(
    entry_point="sentiment_training.py",
    role = 'arn:aws:iam::431525178180:role/hwk4-sagemaker',
    train_instance_type='ml.p2.xlarge',
    train_instance_count=1,
    # output_path='s3://ai2020/hwk4/sagemaker_data/output',
    # framework_version='1.14',
    py_version="py3"
)

estimator.fit({'train': 's3://ai2020/hwk4/sagemaker_data/train/'})

# predictor = estimator.deploy(
#     instance_type='local',
#     initial_instance_count=1
#     )