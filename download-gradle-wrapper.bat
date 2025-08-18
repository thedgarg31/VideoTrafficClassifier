@echo off
echo Downloading Gradle wrapper...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/gradle/gradle/raw/v8.12.0/gradle/wrapper/gradle-wrapper.jar' -OutFile 'gradle\wrapper\gradle-wrapper.jar'"
echo Gradle wrapper downloaded successfully!
