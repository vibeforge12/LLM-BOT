FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES 1

#
# Locale Setting
#
ENV LC_ALL C.UTF-8
RUN echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc

#
# Timezone Setting
#
RUN apt-get update && apt-get install -y tzdata
RUN ls /usr/share/zoneinfo && \
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
echo "Asia/Seoul" > /etc/timezone

RUN apt-get update && apt-get install -y python3-pip
#
# Python Packages
#
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ARG ENVIRONMENT

#
# Entrypoint
#
COPY --chmod=755 docker-entrypoint.${ENVIRONMENT}.sh /docker-entrypoint.sh
ENTRYPOINT "/docker-entrypoint.sh"