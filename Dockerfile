FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb-xinerama0 xauth ffmpeg libsm6 libxext6

RUN apt-get install -y qtbase5-dev qt5-qmake
RUN apt-get install -y libxcb-xinerama0

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger

ENV DISPLAY=:0

WORKDIR /app/src

CMD ["python", "main_program.py"]