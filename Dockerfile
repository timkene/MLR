FROM python:3.10

RUN echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "--server.fileWatcherType=poll", "--browser.serverAddress=0.0.0.0", "--logger.level=error"]
CMD ["Test.py"]
