FROM ubuntu
LABEL maintainer="Carlos Nunez <dev@carlosnunez.me>"

USER root
COPY ./web-server.bash /

RUN chmod 755 /web-server.bash
RUN apt -y update
RUN apt-get update && apt-get install -y bash netcat-openbsd && rm -rf /var/lib/apt/lists/*


USER nobody

ENTRYPOINT [ "/web-server.bash" ]
