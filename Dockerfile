FROM python:3.12-slim

# システムの依存ライブラリをインストール（元の構成を踏襲）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    swig \
    libopenbabel-dev \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# ユーザー権限の設定（セキュリティおよびコンテナ運用のベストプラクティス）
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app/src

# ソースコードとrequirements.txtのコピー
COPY --chown=user . $HOME/app/src

# requirements.txt から不要なモデルのパッケージを除外して軽量化
# （ORBは orb-models が要るので残す。MACE, FairChem, SevenNet, MatterSim等を弾く）
RUN grep -vE "fairchem-core|mace-torch|sevenn|mattersim" requirements.txt > requirements_minimal.txt

# ベースの依存パッケージとORBをインストール
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements_minimal.txt

# PET系（UPET）のインストール（元のコマンドを踏襲）
RUN pip3 install --no-deps upet@git+https://github.com/lab-cosmo/upet.git --ignore-requires-python

# graph_electrostaticsが必要な場合は残す
RUN pip3 install --no-deps git+https://github.com/WillBaldwin0/graph_electrostatics.git --ignore-requires-python

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none"]
