FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update && apt-get install -y libsndfile1-dev vim git p7zip-full && pip install numpy torch librosa==0.7.2 numba==0.48 matplotlib unidecode inflect nltk sndfile tensorboard
