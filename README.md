# Launch Text Preprocessing System:

1. Pull image from Docker Hub:

```
docker pull mariaponomarenko/nlp-system
```

2. Run the following command in CLI:

xhost +local:docker

2. Run container specifying PATH to the files:

```
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v PATH:/src mariaponomarenko/nlp-system
```