# 使用轻量级的 Python 镜像作为基础镜像
FROM python:3.9

# 安装系统依赖（libgl1）以支持 OpenCV
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 到容器中
COPY requirements.txt .

# 安装依赖库
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录下的所有文件到容器的工作目录
COPY . /app

# 暴露端口 5000
EXPOSE 5000

# 运行 Flask 应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]