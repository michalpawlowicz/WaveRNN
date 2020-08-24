FROM tensorflow\tensorflow:latest-gpu
RUN apt-get update && apt-get install -y /
	 libsndfile1-dev \
	 git \
	 vim \
	&& cd ml/WaveRNN \
	&& pip install -r requirements.txt
