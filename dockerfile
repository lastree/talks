FROM jupyter/datascience-notebook


# Use user root
USER root

RUN ls -la ~/
RUN chmod -R 1777 ~/.local
RUN ls -la ~/

