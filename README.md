# Launch Text Processing System:

1. Pull image from Docker Hub:

```
docker pull mariaponomarenko/text-system
```

2. Run the following command in CLI:
```
xhost +local:docker
```
2. Run container specifying PATH to the files:

```
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v PATH:/src mariaponomarenko/text-system
```
