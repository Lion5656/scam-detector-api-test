# 建立docker image(鏡像)

# 選擇base image
FROM python:3.10-slim

# 設定work directory
WORKDIR app

# 安裝套件依賴
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 搬入原始碼
COPY . .

# 執行cmd
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]


