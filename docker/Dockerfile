ARG PY_ABI
FROM quay.io/pypa/manylinux2014_x86_64
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN source $HOME/.cargo/env
RUN git clone https://github.com/tmtenbrink/rustfrc.git opt/rustfrc
RUN alias python3="/opt/python/$PY_ABI/bin/python3"
RUN alias pip3="/opt/python/$PY_ABI/bin/pip3"
RUN pip3 install virtualenv
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
RUN export PATH="/root/.local/bin:$PATH"
RUN cd /opt/rustfrc
RUN poetry update