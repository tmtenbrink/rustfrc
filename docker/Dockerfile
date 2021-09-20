FROM quay.io/pypa/manylinux2014_x86_64
ARG PY_ABI
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN "/opt/python/${PY_ABI}/bin/pip3" install virtualenv
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | "/opt/python/${PY_ABI}/bin/python3" -
RUN echo "export PATH=/root/.local/bin:${PATH}" >> /root/.bashrc
RUN git clone https://github.com/tmtenbrink/rustfrc.git opt/rustfrc
WORKDIR "/opt/rustfrc"
RUN /root/.local/bin/poetry update